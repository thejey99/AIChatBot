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

// Automatic memory sweep: a conversation qualifies when idle this long AND
// has new activity since its last extraction watermark.
const MEMORY_IDLE_MS = 30 * 60 * 1000;
const SWEEP_LIMIT = 2; // max extractions per sweep, bounds cost + latency

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
}

interface ConversationContext {
  systemBlocks: string[];
  history: ChatMessage[];
}

interface SearchSource {
  title: string;
  url: string;
}

// ---------- Firestore helpers ----------

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

    // Messages with an attached image become multipart content so the
    // model can keep seeing the image in follow-up turns.
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
    }));
  facts.sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  return facts;
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

// ---------- Web search (Tavily) ----------

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

// ---------- Memory extraction (shared by button + auto-sweep) ----------

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
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
  try {
    const json = JSON.parse(cleaned);
    if (!Array.isArray(json.new_facts) || !Array.isArray(json.deactivate_ids)) return null;
    return {
      new_facts: json.new_facts.filter((f: unknown) => typeof f === "string" && f.trim()),
      deactivate_ids: json.deactivate_ids.filter((i: unknown) => typeof i === "string"),
    };
  } catch {
    return null;
  }
}

/**
 * Run memory extraction for one conversation, on behalf of its owner.
 * Stamps memoryExtractedAt so the sweep never re-processes an unchanged
 * conversation. Returns null on failure (harmless; retried next sweep).
 */
async function extractConversationMemory(
  user: AuthedUser,
  conversationId: string,
  log: any
): Promise<{ added: string[]; deactivated: number } | null> {
  const convRef = conversations.doc(conversationId);

  const msgSnap = await convRef.collection("messages").orderBy("createdAt", "asc").get();
  if (msgSnap.empty) return { added: [], deactivated: 0 };

  const transcript = msgSnap.docs
    .map((d) => {
      const who = d.data().role === "user" ? "User" : "Assistant";
      const img = d.data().image ? " [image attached]" : "";
      return who + ":" + img + " " + d.data().content;
    })
    .join("\n");

  const existing = await loadActiveFacts(user);
  const existingBlock =
    existing.length > 0
      ? existing.map((f) => "[" + f.id + "] " + f.text).join("\n")
      : "(none)";

  const extractionMessages: ChatMessage[] = [
    { role: "system", content: EXTRACTION_INSTRUCTIONS },
    {
      role: "user",
      content: "EXISTING FACTS:\n" + existingBlock + "\n\nTRANSCRIPT:\n" + transcript,
    },
  ];

  let result: ExtractionResult | null = null;
  for (let attempt = 0; attempt < 2 && !result; attempt++) {
    try {
      const raw = await llmComplete(extractionMessages);
      result = parseExtraction(raw);
    } catch (err) {
      log.error(err, "extraction call failed");
    }
  }

  if (!result) return null;

  const validIds = new Set(existing.map((f) => f.id));
  const batch = db.batch();

  let deactivated = 0;
  for (const id of result.deactivate_ids) {
    if (!validIds.has(id)) continue;
    batch.update(memory.doc(id), { active: false });
    deactivated++;
  }

  for (const text of result.new_facts) {
    batch.set(memory.doc(), {
      uid: user.uid,
      text: text.trim(),
      active: true,
      createdAt: FieldValue.serverTimestamp(),
      sourceConversationId: conversationId,
    });
  }

  // Watermark: this conversation is extracted as of now
  batch.update(convRef, { memoryExtractedAt: FieldValue.serverTimestamp() });

  await batch.commit();

  log.info(
    { conversationId, added: result.new_facts.length, deactivated, user: user.email },
    "memory extraction applied"
  );

  return { added: result.new_facts, deactivated };
}

// ---------- Automatic memory sweep ----------

async function sweepIdleConversations(user: AuthedUser, log: any): Promise<void> {
  const now = Date.now();
  const snap = await conversations.orderBy("updatedAt", "desc").limit(100).get();

  const candidates = snap.docs.filter((d) => {
    if (!ownsDoc(user, d.data())) return false;
    const updatedAt = tsToMillis(d.data().updatedAt);
    if (!updatedAt) return false;
    if (now - updatedAt < MEMORY_IDLE_MS) return false; // still active
    const extractedAt = tsToMillis(d.data().memoryExtractedAt);
    return !extractedAt || extractedAt < updatedAt; // new activity since last sweep
  });

  for (const doc of candidates.slice(0, SWEEP_LIMIT)) {
    try {
      await extractConversationMemory(user, doc.id, log);
    } catch (err) {
      log.error({ conversationId: doc.id, err }, "auto memory sweep failed");
    }
  }
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

  const doc = await memory.add({
    uid: user.uid,
    text,
    active: true,
    createdAt: FieldValue.serverTimestamp(),
    sourceConversationId: null,
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

// ---------- Route: manual extraction ("Remember" button) ----------

app.post<{ Params: { id: string } }>(
  "/api/conversations/:id/remember",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const conversationId = request.params.id;
    const convRef = conversations.doc(conversationId);
    const convSnap = await convRef.get();
    if (!convSnap.exists || !ownsDoc(user, convSnap.data())) {
      return reply.code(404).send({ error: "Not found" });
    }

    const result = await extractConversationMemory(user, conversationId, request.log);
    if (!result) {
      return reply.code(502).send({ error: "Extraction failed after retry" });
    }
    return result;
  }
);

// ---------- Route: chat ----------

interface StreamedToolCall {
  id: string;
  name: string;
  arguments: string;
}

async function streamOneRound(
  provider: ProviderConfig,
  messages: ChatMessage[],
  offerTools: boolean,
  emit: (frame: object) => void,
  log: any
): Promise<{ content: string; toolCalls: StreamedToolCall[]; failed: boolean }> {
  const body: Record<string, unknown> = {
    model: provider.model,
    messages,
    stream: true,
  };
  if (offerTools) body.tools = [SEARCH_TOOL];

  const upstream = await fetch(provider.baseUrl + "/chat/completions", {
    method: "POST",
    headers: providerHeaders(provider),
    body: JSON.stringify(body),
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    log.error({ status: upstream.status, model: provider.model, detail }, "LLM provider error");
    return { content: "", toolCalls: [], failed: true };
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let lineBuffer = "";
  let content = "";
  const toolCalls: Record<number, StreamedToolCall> = {};

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    lineBuffer += decoder.decode(value, { stream: true });
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") continue;

      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta;
        if (!delta) continue;

        if (typeof delta.content === "string" && delta.content) {
          content += delta.content;
          emit({ choices: [{ delta: { content: delta.content } }] });
        }

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index ?? 0;
            if (!toolCalls[idx]) {
              toolCalls[idx] = { id: tc.id ?? "call_" + idx, name: "", arguments: "" };
            }
            if (tc.id) toolCalls[idx].id = tc.id;
            if (tc.function?.name) toolCalls[idx].name += tc.function.name;
            if (tc.function?.arguments) toolCalls[idx].arguments += tc.function.arguments;
          }
        }
      } catch {
        // ignore partial frames
      }
    }
  }

  return { content, toolCalls: Object.values(toolCalls), failed: false };
}

app.post<{ Body: ChatBody }>("/api/chat", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const body = request.body ?? {};
  const baseSystem =
    body.system ?? "You are a helpful personal assistant. Be concise and direct.";

  // Validate image payload if present
  let image: string | undefined;
  if (typeof body.image === "string" && body.image) {
    if (!body.image.startsWith("data:image/")) {
      return reply.code(400).send({ error: "image must be a data:image/* URL" });
    }
    if (body.image.length > MAX_IMAGE_BYTES) {
      return reply.code(400).send({ error: "Image too large (max ~900KB after compression)" });
    }
    image = body.image;
  }

  // Images force the multimodal default lane; PRO (Groq) is text-only.
  const provider = image
    ? DEFAULT_PROVIDER
    : MODEL_ALIASES[body.model ?? "default"] ?? DEFAULT_PROVIDER;

  const facts = await loadActiveFacts(user);

  let workingMessages: ChatMessage[];
  let conversationId: string | null = null;

  if (typeof body.message === "string" && (body.message.trim() || image)) {
    const userText = body.message.trim();

    if (body.conversationId) {
      const convRef = conversations.doc(body.conversationId);
      const convSnap = await convRef.get();
      if (!convSnap.exists || !ownsDoc(user, convSnap.data())) {
        return reply.code(404).send({ error: "Conversation not found" });
      }
      conversationId = body.conversationId;
      if (!convSnap.data()?.uid) await convRef.update({ uid: user.uid });
    } else {
      const doc = await conversations.add({
        uid: user.uid,
        title: (userText || "Image").slice(0, 60),
        pinned: false,
        createdAt: FieldValue.serverTimestamp(),
        updatedAt: FieldValue.serverTimestamp(),
      });
      conversationId = doc.id;
    }

    const convRef = conversations.doc(conversationId);
    await convRef.collection("messages").add({
      role: "user",
      content: userText,
      ...(image ? { image } : {}),
      createdAt: FieldValue.serverTimestamp(),
    });

    const context = await loadConversationContext(conversationId);
    const systemPrompt = buildSystemPrompt(baseSystem, facts, context.systemBlocks);
    workingMessages = [{ role: "system", content: systemPrompt }, ...context.history];
  } else if (Array.isArray(body.messages) && body.messages.length > 0) {
    const systemPrompt = buildSystemPrompt(baseSystem, facts, []);
    workingMessages = [
      { role: "system", content: systemPrompt },
      ...body.messages.filter((m) => m.role === "user" || m.role === "assistant"),
    ];
  } else {
    return reply.code(400).send({ error: "Provide 'message' or 'messages'" });
  }

  reply.raw.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Expose-Headers": "X-Conversation-Id",
    ...(conversationId ? { "X-Conversation-Id": conversationId } : {}),
  });

  const emit = (frame: object) => {
    reply.raw.write("data: " + JSON.stringify(frame) + "\n\n");
  };

  let assistantText = "";
  let searchesUsed = 0;
  const collectedSources: SearchSource[] = [];
  const seenUrls = new Set<string>();

  try {
    for (let round = 0; round <= MAX_SEARCH_ROUNDS; round++) {
      const offerTools = Boolean(TAVILY_API_KEY) && round < MAX_SEARCH_ROUNDS;
      const result = await streamOneRound(
        provider,
        workingMessages,
        offerTools,
        emit,
        request.log
      );

      if (result.failed) {
        emit({
          choices: [
            {
              delta: {
                content:
                  assistantText.length > 0
                    ? "\n\n*(The model hit an error while finishing this answer.)*"
                    : "The model provider returned an error. If you were using Pro, try switching back to fast.",
              },
            },
          ],
        });
        break;
      }

      assistantText += result.content;

      if (result.toolCalls.length === 0) break;

      workingMessages.push({
        role: "assistant",
        content: result.content || null,
        tool_calls: result.toolCalls.map((tc) => ({
          id: tc.id,
          type: "function" as const,
          function: { name: tc.name, arguments: tc.arguments },
        })),
      });

      for (const tc of result.toolCalls) {
        let query = "";
        try {
          query = String(JSON.parse(tc.arguments || "{}").query ?? "");
        } catch {
          query = "";
        }

        emit({ search: { query: query || "(unspecified)" } });
        searchesUsed++;

        let toolText = "Invalid search arguments. Answer from your own knowledge.";
        if (query) {
          const { text, sources } = await tavilySearch(query, request.log);
          toolText = text;
          for (const s of sources) {
            if (!seenUrls.has(s.url)) {
              seenUrls.add(s.url);
              collectedSources.push(s);
            }
          }
        }

        workingMessages.push({
          role: "tool",
          content: toolText,
          tool_call_id: tc.id,
        });
      }
    }
  } catch (err) {
    request.log.error(err, "chat loop failed");
  } finally {
    if (collectedSources.length > 0) {
      emit({ sources: collectedSources });
    }
    reply.raw.write("data: [DONE]\n\n");
    reply.raw.end();
  }

  if (conversationId && assistantText) {
    const convRef = conversations.doc(conversationId);
    await convRef.collection("messages").add({
      role: "assistant",
      content: assistantText,
      ...(collectedSources.length > 0 ? { sources: collectedSources } : {}),
      createdAt: FieldValue.serverTimestamp(),
    });
    await convRef.update({ updatedAt: FieldValue.serverTimestamp() });

    try {
      await maybeCompact(conversationId, request.log);
    } catch (err) {
      request.log.error(err, "compaction failed (will retry next turn)");
    }

    try {
      await sweepIdleConversations(user, request.log);
    } catch (err) {
      request.log.error(err, "memory sweep failed (will retry next request)");
    }
  }

  request.log.info(
    {
      user: user.email,
      conversationId,
      model: provider.model,
      searches: searchesUsed,
      hasImage: Boolean(image),
    },
    "chat completed"
  );
});

// ---------- Start ----------

app.listen({ port: PORT, host: "0.0.0.0" }).catch((err) => {
  app.log.error(err);
  process.exit(1);
});
