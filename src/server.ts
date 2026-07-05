import Fastify from "fastify";
import cors from "@fastify/cors";
import { readFileSync } from "node:fs";
import { initializeApp, cert } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore, FieldValue, Timestamp } from "firebase-admin/firestore";

// ---------- Configuration ----------

const PORT = Number(process.env.PORT ?? 8080);

const BOOTSTRAP_ADMINS = (
  process.env.BOOTSTRAP_ADMINS ??
  "jryan@charlestownehotels.com,john99ran@gmail.com"
)
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

const LEGACY_OWNER_EMAIL = (process.env.LEGACY_OWNER_EMAIL ?? "john99ran@gmail.com")
  .trim()
  .toLowerCase();

const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";

const LLM_BASE_URL =
  process.env.LLM_BASE_URL ??
  "https://generativelanguage.googleapis.com/v1beta/openai";
const LLM_MODEL = process.env.LLM_MODEL ?? "gemini-2.5-flash";
const LLM_API_KEY = process.env.LLM_API_KEY ?? "";

const PRO_BASE_URL = process.env.PRO_BASE_URL ?? "https://api.groq.com/openai/v1";
const PRO_MODEL = process.env.PRO_MODEL ?? "openai/gpt-oss-120b";
const PRO_API_KEY = process.env.PRO_API_KEY ?? "";

// Web search (Tavily). If unset, the search tool is not offered at all.
const TAVILY_API_KEY = process.env.TAVILY_API_KEY ?? "";
const MAX_SEARCH_ROUNDS = 3;

// Max accepted image payload (base64 data URL). Firestore docs cap at 1MB.
const MAX_IMAGE_BYTES = 900_000;

// Code-fence marker (three backticks), built dynamically so this source file
// never contains literal backticks (they corrupt when shared via markdown).
const FENCE_HINT = String.fromCharCode(96, 96, 96);

interface ProviderConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

const DEFAULT_PROVIDER: ProviderConfig = {
  baseUrl: LLM_BASE_URL,
  apiKey: LLM_API_KEY,
  model: LLM_MODEL,
};

const PRO_PROVIDER: ProviderConfig = PRO_API_KEY
  ? { baseUrl: PRO_BASE_URL, apiKey: PRO_API_KEY, model: PRO_MODEL }
  : DEFAULT_PROVIDER;

const MODEL_ALIASES: Record<string, ProviderConfig> = {
  default: DEFAULT_PROVIDER,
  pro: PRO_PROVIDER,
};

const COMPACT_TRIGGER = 50;
const COMPACT_KEEP = 20;

const SECRET_FILE_PATH = "/etc/secrets/firebase-service-account.json";

function loadServiceAccount(): Record<string, unknown> {
  try {
    const raw = readFileSync(SECRET_FILE_PATH, "utf8");
    console.log("Firebase credentials loaded from secret file");
    return JSON.parse(raw);
  } catch {
    console.log("Secret file not found, falling back to FIREBASE_SERVICE_ACCOUNT env var");
    try {
      return JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT ?? "");
    } catch {
      console.error(
        "FATAL: No valid Firebase credentials. Add the Secret File " +
          "'firebase-service-account.json' in Render's Environment tab."
      );
      process.exit(1);
    }
  }
}

initializeApp({ credential: cert(loadServiceAccount()) });
const db = getFirestore();

// ---------- Server ----------

const app = Fastify({
  logger: true,
  bodyLimit: 2 * 1024 * 1024, // allow image payloads (default is 1MB)
});
await app.register(cors, {
  origin: ALLOWED_ORIGIN,
  exposedHeaders: ["X-Conversation-Id"],
});

// ---------- Collections ----------

const conversations = db.collection("conversations");
const memory = db.collection("memory");
const allowlist = db.collection("allowlist");

// ---------- Allowlist bootstrap ----------

async function seedAllowlist(): Promise<void> {
  for (const email of BOOTSTRAP_ADMINS) {
    const ref = allowlist.doc(email);
    const snap = await ref.get();
    if (!snap.exists) {
      await ref.set({
        email,
        role: "admin",
        addedAt: FieldValue.serverTimestamp(),
        addedBy: "bootstrap",
      });
      console.log("Seeded bootstrap admin: " + email);
    } else if (snap.data()?.role !== "admin") {
      await ref.update({ role: "admin" });
    }
  }
}
await seedAllowlist();

// ---------- Auth ----------

interface AuthedUser {
  uid: string;
  email: string;
  role: "admin" | "user";
}

async function requireAllowedUser(authHeader: string | undefined): Promise<AuthedUser> {
  if (!authHeader?.startsWith("Bearer ")) {
    throw { statusCode: 401, message: "Missing bearer token" };
  }
  const idToken = authHeader.slice(7);

  const decoded = await getAuth()
    .verifyIdToken(idToken)
    .catch(() => {
      throw { statusCode: 401, message: "Invalid or expired token" };
    });

  const email = decoded.email?.toLowerCase();
  if (!email || !decoded.email_verified) {
    throw { statusCode: 403, message: "Email not verified", attemptedEmail: email };
  }

  const entry = await allowlist.doc(email).get();
  if (!entry.exists) {
    throw {
      statusCode: 403,
      message: "This account is not authorized. Ask an admin to add you.",
      attemptedEmail: email,
    };
  }

  const role = (entry.data()?.role as "admin" | "user") ?? "user";
  return { uid: decoded.uid, email, role };
}

async function guard(request: any, reply: any): Promise<AuthedUser | null> {
  try {
    return await requireAllowedUser(request.headers.authorization);
  } catch (err: any) {
    request.log.warn(
      { attemptedEmail: err.attemptedEmail ?? "unknown", reason: err.message },
      "auth rejected"
    );
    reply.code(err.statusCode ?? 401).send({ error: err.message });
    return null;
  }
}

async function guardAdmin(request: any, reply: any): Promise<AuthedUser | null> {
  const user = await guard(request, reply);
  if (!user) return null;
  if (user.role !== "admin") {
    reply.code(403).send({ error: "Admin access required" });
    return null;
  }
  return user;
}

// ---------- Ownership ----------

function ownsDoc(
  user: AuthedUser,
  data: FirebaseFirestore.DocumentData | undefined
): boolean {
  if (!data) return false;
  const docUid = data.uid as string | undefined;
  if (docUid) return docUid === user.uid;
  return user.email === LEGACY_OWNER_EMAIL;
}

// ---------- Types ----------

type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | ContentPart[] | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

interface ChatBody {
  message?: string;
  conversationId?: string;
  messages?: ChatMessage[];
  system?: string;
  model?: string;
  image?: string; // data URL, image/* only
}

interface MemoryFact {
  id: string;
  text: string;
  active: boolean;
  createdAt: number | null;
  sourceConversationId: string | null;
  embedding?: number[];
}

interface ConversationContext {
  systemBlocks: string[];
  history: ChatMessage[];
}

interface SearchSource {
  title: string;
  url: string;
}

// ---------- Firestore & RAG Helpers ----------

function tsToMillis(v: unknown): number | null {
  return v instanceof Timestamp ? v.toMillis() : null;
}

async function loadConversationContext(
  conversationId: string
): Promise<ConversationContext> {
  const convSnap = await conversations.doc(conversationId).get();
  const summary: string = convSnap.data()?.summary ?? "";
  const summaryUpTo: Timestamp | null = convSnap.data()?.summaryUpTo ?? null;

  let query = conversations
    .doc(conversationId)
    .collection("messages")
    .orderBy("createdAt", "asc");

  if (summaryUpTo) {
    query = query.where("createdAt", ">", summaryUpTo);
  }

  const snap = await query.get();
  const history: ChatMessage[] = snap.docs.map((d) => {
    const role = d.data().role as "user" | "assistant";
    const text = String(d.data().content ?? "");
    const image = d.data().image as string | undefined;

    if (image && role === "user") {
      const parts: ContentPart[] = [];
      if (text) parts.push({ type: "text", text });
      parts.push({ type: "image_url", image_url: { url: image } });
      return { role, content: parts };
    }
    return { role, content: text };
  });

  const systemBlocks: string[] = [];
  if (summary) {
    systemBlocks.push(
      "Summary of the earlier part of this conversation (older messages have been condensed):\n" +
        summary
    );
  }

  return { systemBlocks, history };
}

async function loadActiveFacts(user: AuthedUser): Promise<MemoryFact[]> {
  const snap = await memory.where("active", "==", true).get();
  const facts = snap.docs
    .filter((d) => ownsDoc(user, d.data()))
    .map((d) => ({
      id: d.id,
      text: String(d.data().text ?? ""),
      active: true,
      createdAt: tsToMillis(d.data().createdAt),
      sourceConversationId: d.data().sourceConversationId ?? null,
      embedding: d.data().embedding ?? [],
    }));
  facts.sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  return facts;
}

async function getEmbedding(text: string): Promise<number[]> {
  try {
    const res = await fetch(DEFAULT_PROVIDER.baseUrl + "/embeddings", {
      method: "POST",
      headers: providerHeaders(DEFAULT_PROVIDER),
      body: JSON.stringify({
        model: "text-embedding-3-small", 
        input: text,
      }),
    });
    if (!res.ok) return [];
    const json = await res.json();
    return json.data?.[0]?.embedding || [];
  } catch {
    return []; // Fallback gracefully if provider lacks an embedding endpoint
  }
}

function cosineSimilarity(vecA: number[], vecB: number[]) {
  if (!vecA?.length || !vecB?.length || vecA.length !== vecB.length) return 0;
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function getRelevantFacts(user: AuthedUser, queryText: string): Promise<MemoryFact[]> {
  const allFacts = await loadActiveFacts(user);
  if (allFacts.length === 0 || !queryText) return allFacts.slice(-10); // fallback latest 10

  const queryEmbedding = await getEmbedding(queryText);
  if (queryEmbedding.length === 0) return allFacts.slice(-10); // fallback

  const scored = allFacts.map(fact => ({
    ...fact,
    score: fact.embedding?.length ? cosineSimilarity(queryEmbedding, fact.embedding) : 0,
  }));

  // Sort by highest similarity
  scored.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  return scored.slice(0, 5); // Return top 5 most relevant facts to save context
}

function buildSystemPrompt(
  baseSystem: string,
  facts: MemoryFact[],
  extraBlocks: string[]
): string {
  const parts = [baseSystem];

  if (TAVILY_API_KEY) {
    parts.push(
      "You have a web_search tool. Use it when the question involves current events, " +
        "recent information, prices, weather, news, or anything you are unsure is up to date. " +
        "Do not use it for stable knowledge, personal conversation, or things already in context. " +
        "When you use search results, mention your sources briefly."
    );
  }

  parts.push(
    "When the user asks for an interactive demo, game, or runnable code, produce one complete, " +
      "self-contained HTML file in a single " + FENCE_HINT + "html code block (inline CSS and JS, no external files). " +
      "Make it work on BOTH desktop and mobile: support touch events (touchstart/touchmove) alongside mouse events, " +
      "size the canvas responsively to the container (window.innerWidth/innerHeight, handle resize), " +
      "and use large touch-friendly controls. The user runs it directly in a sandboxed iframe."
  );

  if (facts.length > 0) {
    const factLines = facts.map((f) => "- " + f.text).join("\n");
    parts.push(
      "You have persistent memory of the user from previous conversations. " +
        "Use these facts naturally when relevant; do not recite them unprompted:\n" +
        factLines
    );
  }

  parts.push(...extraBlocks);
  return parts.join("\n\n");
}

// ---------- LLM helpers ----------

function providerHeaders(provider: ProviderConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: "Bearer " + provider.apiKey,
  };
  if (provider.baseUrl.includes("openrouter.ai")) {
    headers["X-Title"] = "Personal AI Chat";
  }
  return headers;
}

async function llmComplete(messages: ChatMessage[]): Promise<string> {
  const res = await fetch(DEFAULT_PROVIDER.baseUrl + "/chat/completions", {
    method: "POST",
    headers: providerHeaders(DEFAULT_PROVIDER),
    body: JSON.stringify({ model: DEFAULT_PROVIDER.model, messages, stream: false }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error("LLM error " + res.status + ": " + detail.slice(0, 300));
  }
  const json = await res.json();
  return String(json.choices?.[0]?.message?.content ?? "");
}

// ---------- Tools & Web search (Tavily) ----------

const SEARCH_TOOL = {
  type: "function" as const,
  function: {
    name: "web_search",
    description:
      "Search the web for current, real-time, or recent information. " +
      "Use for news, prices, weather, sports, recent releases, or anything that may have changed recently.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "A concise search query, 2-6 words.",
        },
      },
      required: ["query"],
    },
  },
};

const FETCH_URL_TOOL = {
  type: "function" as const,
  function: {
    name: "fetch_url",
    description: "Fetch and read the text content of a given URL. Useful for reading articles, documentation, or links provided by the user.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "The full URL to fetch (e.g., https://example.com)" },
      },
      required: ["url"],
    },
  },
};

const TIME_TOOL = {
  type: "function" as const,
  function: {
    name: "get_current_time",
    description: "Get the exact current date and time.",
    parameters: { type: "object", properties: {} },
  },
};

async function tavilySearch(
  query: string,
  log: any
): Promise<{ text: string; sources: SearchSource[] }> {
  try {
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + TAVILY_API_KEY,
      },
      body: JSON.stringify({
        query,
        max_results: 5,
        include_answer: true,
        search_depth: "basic",
      }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      log.error({ status: res.status, detail: detail.slice(0, 200) }, "Tavily error");
      return {
        text:
          "Search failed (" +
          res.status +
          "). Answer from your own knowledge and say you could not verify current information.",
        sources: [],
      };
    }

    const json = await res.json();
    const parts: string[] = [];
    const sources: SearchSource[] = [];

    if (json.answer) parts.push("Summary: " + json.answer);
    for (const r of json.results ?? []) {
      parts.push(
        "- " + r.title + " (" + r.url + ")\n  " + String(r.content ?? "").slice(0, 400)
      );
      if (r.url) sources.push({ title: String(r.title ?? r.url), url: String(r.url) });
    }

    return {
      text: parts.length > 0 ? parts.join("\n") : "No results found.",
      sources,
    };
  } catch (err) {
    log.error(err, "Tavily request failed");
    return {
      text: "Search failed. Answer from your own knowledge and say you could not verify current information.",
      sources: [],
    };
  }
}

// ---------- Compaction ----------

const COMPACTION_INSTRUCTIONS =
  "You maintain a rolling summary of a conversation between a user and an assistant.\n\n" +
  "You will receive the PREVIOUS SUMMARY (possibly empty) and a batch of OLDER MESSAGES that must now be folded into it.\n\n" +
  "Write an updated summary that:\n" +
  "- Preserves facts, decisions, preferences, open questions, and anything either party may refer back to later\n" +
  "- Preserves specific names, numbers, code identifiers, and URLs mentioned\n" +
  "- Drops pleasantries and redundancy\n" +
  "- Is written in compact plain prose, at most ~400 words\n\n" +
  "Respond with ONLY the updated summary text.";

async function maybeCompact(conversationId: string, log: any): Promise<void> {
  const convRef = conversations.doc(conversationId);
  const convSnap = await convRef.get();
  const prevSummary: string = convSnap.data()?.summary ?? "";
  const summaryUpTo: Timestamp | null = convSnap.data()?.summaryUpTo ?? null;

  let tailQuery = convRef.collection("messages").orderBy("createdAt", "asc");
  if (summaryUpTo) {
    tailQuery = tailQuery.where("createdAt", ">", summaryUpTo);
  }

  const tailSnap = await tailQuery.get();
  if (tailSnap.size <= COMPACT_TRIGGER) return;

  const toFold = tailSnap.docs.slice(0, tailSnap.size - COMPACT_KEEP);
  if (toFold.length === 0) return;

  const batchText = toFold
    .map((d) => {
      const who = d.data().role === "user" ? "User" : "Assistant";
      const img = d.data().image ? " [image attached]" : "";
      return who + ":" + img + " " + d.data().content;
    })
    .join("\n");

  const updatedSummary = await llmComplete([
    { role: "system", content: COMPACTION_INSTRUCTIONS },
    {
      role: "user",
      content:
        "PREVIOUS SUMMARY:\n" +
        (prevSummary || "(empty)") +
        "\n\nOLDER MESSAGES:\n" +
        batchText,
    },
  ]);

  if (!updatedSummary.trim()) {
    log.warn({ conversationId }, "compaction produced empty summary; skipping");
    return;
  }

  const lastFolded = toFold[toFold.length - 1];
  await convRef.update({
    summary: updatedSummary.trim(),
    summaryUpTo: lastFolded.data().createdAt,
  });

  log.info({ conversationId, folded: toFold.length }, "conversation compacted");
}

// ---------- Routes: health & me ----------

app.get("/api/health", async () => ({ ok: true }));

app.get("/api/me", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;
  return { email: user.email, role: user.role };
});

// ---------- Routes: allowlist (admin only) ----------

app.get("/api/allowlist", async (request, reply) => {
  const user = await guardAdmin(request, reply);
  if (!user) return;

  const snap = await allowlist.get();
  const entries = snap.docs.map((d) => ({
    email: d.id,
    role: d.data().role ?? "user",
    addedAt: tsToMillis(d.data().addedAt),
  }));
  entries.sort((a, b) => (b.addedAt ?? 0) - (a.addedAt ?? 0));
  return entries;
});

app.post<{ Body: { email?: string; role?: string } }>(
  "/api/allowlist",
  async (request, reply) => {
    const user = await guardAdmin(request, reply);
    if (!user) return;

    const email = request.body?.email?.trim().toLowerCase();
    const role = request.body?.role === "admin" ? "admin" : "user";
    if (!email || !email.includes("@")) {
      return reply.code(400).send({ error: "Valid email required" });
    }

    await allowlist.doc(email).set({
      email,
      role,
      addedAt: FieldValue.serverTimestamp(),
      addedBy: user.email,
    });
    return { ok: true };
  }
);

app.delete<{ Params: { email: string } }>(
  "/api/allowlist/:email",
  async (request, reply) => {
    const user = await guardAdmin(request, reply);
    if (!user) return;

    const target = decodeURIComponent(request.params.email).toLowerCase();
    if (target === user.email) {
      return reply.code(400).send({ error: "You cannot remove yourself" });
    }
    await allowlist.doc(target).delete();
    return { ok: true };
  }
);

// ---------- Routes: conversations ----------

app.get("/api/conversations", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const snap = await conversations.orderBy("updatedAt", "desc").limit(200).get();
  const mine = snap.docs
    .filter((d) => ownsDoc(user, d.data()))
    .map((d) => ({
      id: d.id,
      title: d.data().title ?? "Untitled",
      updatedAt: tsToMillis(d.data().updatedAt),
      pinned: d.data().pinned === true,
    }));

  mine.sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    return (b.updatedAt ?? 0) - (a.updatedAt ?? 0);
  });

  return mine.slice(0, 50);
});

app.get<{ Params: { id: string } }>(
  "/api/conversations/:id/messages",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const convRef = conversations.doc(request.params.id);
    const convSnap = await convRef.get();
    if (!convSnap.exists || !ownsDoc(user, convSnap.data())) {
      return reply.code(404).send({ error: "Not found" });
    }

    const snap = await convRef.collection("messages").orderBy("createdAt", "asc").get();
    return snap.docs.map((d) => ({
      role: d.data().role,
      content: d.data().content,
      image: d.data().image ?? null,
      sources: d.data().sources ?? null,
      createdAt: tsToMillis(d.data().createdAt),
    }));
  }
);

app.patch<{ Params: { id: string }; Body: { pinned?: boolean } }>(
  "/api/conversations/:id",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const convRef = conversations.doc(request.params.id);
    const convSnap = await convRef.get();
    if (!convSnap.exists || !ownsDoc(user, convSnap.data())) {
      return reply.code(404).send({ error: "Not found" });
    }

    const updates: Record<string, unknown> = {};
    if (typeof request.body?.pinned === "boolean") updates.pinned = request.body.pinned;
    if (!convSnap.data()?.uid) updates.uid = user.uid;

    if (Object.keys(updates).length === 0) {
      return reply.code(400).send({ error: "Nothing to update" });
    }
    await convRef.update(updates);
    return { ok: true };
  }
);

app.delete<{ Params: { id: string } }>(
  "/api/conversations/:id",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const convRef = conversations.doc(request.params.id);
    const convSnap = await convRef.get();
    if (!convSnap.exists || !ownsDoc(user, convSnap.data())) {
      return reply.code(404).send({ error: "Not found" });
    }

    await db.recursiveDelete(convRef);
    return { ok: true };
  }
);

// ---------- Routes: memory ----------

app.get("/api/memory", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const snap = await memory.orderBy("createdAt", "desc").limit(1000).get();
  return snap.docs
    .filter((d) => ownsDoc(user, d.data()))
    .map((d) => ({
      id: d.id,
      text: d.data().text ?? "",
      active: d.data().active !== false,
      createdAt: tsToMillis(d.data().createdAt),
      sourceConversationId: d.data().sourceConversationId ?? null,
    }))
    .slice(0, 500);
});

app.post<{ Body: { text?: string } }>("/api/memory", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const text = request.body?.text?.trim();
  if (!text) return reply.code(400).send({ error: "text required" });

  const embedding = await getEmbedding(text);

  const doc = await memory.add({
    uid: user.uid,
    text,
    active: true,
    createdAt: FieldValue.serverTimestamp(),
    sourceConversationId: null,
    embedding,
  });
  return { id: doc.id };
});

app.patch<{ Params: { id: string }; Body: { text?: string; active?: boolean } }>(
  "/api/memory/:id",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const ref = memory.doc(request.params.id);
    const snap = await ref.get();
    if (!snap.exists || !ownsDoc(user, snap.data())) {
      return reply.code(404).send({ error: "Not found" });
    }

    const updates: Record<string, unknown> = {};
    if (typeof request.body?.text === "string" && request.body.text.trim()) {
      updates.text = request.body.text.trim();
      updates.embedding = await getEmbedding(updates.text as string); // Update embedding when text changes
    }
    if (typeof request.body?.active === "boolean") {
      updates.active = request.body.active;
    }
    if (!snap.data()?.uid) updates.uid = user.uid;
    if (Object.keys(updates).length === 0) {
      return reply.code(400).send({ error: "Nothing to update" });
    }

    await ref.update(updates);
    return { ok: true };
  }
);

// ---------- Extraction Logic (Auto + Manual) ----------

const EXTRACTION_INSTRUCTIONS =
  "You maintain a long-term memory of durable facts about the user.\n\n" +
  "Given the conversation transcript and the list of EXISTING facts, respond with ONLY a JSON object, no markdown fences, in exactly this shape:\n" +
  '{"new_facts": ["..."], "deactivate_ids": ["..."]}\n\n' +
  "Rules:\n" +
  '- new_facts: durable facts about the user worth remembering across future conversations (preferences, projects, people, decisions, circumstances). Write each as one short standalone sentence about "the user".\n' +
  "- Do NOT include facts already covered by an existing fact.\n" +
  "- Do NOT include trivia, small talk, or one-off details with no future value.\n" +
  "- deactivate_ids: IDs of existing facts that this conversation shows are now false, outdated, or superseded.\n" +
  '- If nothing qualifies, return {"new_facts": [], "deactivate_ids": []}.';

interface ExtractionResult {
  new_facts: string[];
  deactivate_ids: string[];
}

function parseExtraction(raw: string): ExtractionResult | null {
  const cleaned = raw
    .trim()
    .replace(/^)
