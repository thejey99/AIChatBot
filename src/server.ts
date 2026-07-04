import Fastify from "fastify";
import cors from "@fastify/cors";
import { readFileSync } from "node:fs";
import { initializeApp, cert } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore, FieldValue, Timestamp } from "firebase-admin/firestore";

// ---------- Configuration ----------

const PORT = Number(process.env.PORT ?? 8080);

const ALLOWED_EMAILS = new Set(
  (process.env.ALLOWED_EMAILS ?? "jryan@charlestownehotels.com,john99ran@gmail.com")
    .split(",")
    .map((e) => e.trim().toLowerCase())
);

const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";

const LLM_BASE_URL =
  process.env.LLM_BASE_URL ??
  "https://generativelanguage.googleapis.com/v1beta/openai";
const LLM_MODEL = process.env.LLM_MODEL ?? "gemini-2.5-flash";
const LLM_API_KEY = process.env.LLM_API_KEY ?? "";

const HISTORY_LIMIT = 30;

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

const app = Fastify({ logger: true });
await app.register(cors, {
  origin: ALLOWED_ORIGIN,
  exposedHeaders: ["X-Conversation-Id"],
});

// ---------- Auth ----------

interface AuthedUser {
  uid: string;
  email: string;
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
  if (!email || !decoded.email_verified || !ALLOWED_EMAILS.has(email)) {
    throw { statusCode: 403, message: "This account is not authorized" };
  }
  return { uid: decoded.uid, email };
}

async function guard(request: any, reply: any): Promise<AuthedUser | null> {
  try {
    return await requireAllowedUser(request.headers.authorization);
  } catch (err: any) {
    reply.code(err.statusCode ?? 401).send({ error: err.message });
    return null;
  }
}

// ---------- Types ----------

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatBody {
  message?: string;
  conversationId?: string;
  messages?: ChatMessage[];
  system?: string;
}

interface MemoryFact {
  id: string;
  text: string;
  active: boolean;
  createdAt: number | null;
  sourceConversationId: string | null;
}

// ---------- Firestore helpers ----------

const conversations = db.collection("conversations");
const memory = db.collection("memory");

function tsToMillis(v: unknown): number | null {
  return v instanceof Timestamp ? v.toMillis() : null;
}

async function loadHistory(conversationId: string): Promise<ChatMessage[]> {
  const snap = await conversations
    .doc(conversationId)
    .collection("messages")
    .orderBy("createdAt", "desc")
    .limit(HISTORY_LIMIT)
    .get();

  return snap.docs
    .map((d) => d.data() as { role: "user" | "assistant"; content: string })
    .reverse()
    .map((m) => ({ role: m.role, content: m.content }));
}

async function loadActiveFacts(): Promise<MemoryFact[]> {
  const snap = await memory.where("active", "==", true).get();
  const facts = snap.docs.map((d) => ({
    id: d.id,
    text: String(d.data().text ?? ""),
    active: true,
    createdAt: tsToMillis(d.data().createdAt),
    sourceConversationId: d.data().sourceConversationId ?? null,
  }));
  facts.sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  return facts;
}

function buildSystemPrompt(baseSystem: string, facts: MemoryFact[]): string {
  if (facts.length === 0) return baseSystem;
  const factLines = facts.map((f) => `- ${f.text}`).join("\n");
  return (
    `${baseSystem}\n\n` +
    `You have persistent memory of the user from previous conversations. ` +
    `Use these facts naturally when relevant; do not recite them unprompted:\n${factLines}`
  );
}

// ---------- LLM helpers ----------

async function llmComplete(messages: ChatMessage[]): Promise<string> {
  const res = await fetch(`${LLM_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LLM_API_KEY}`,
    },
    body: JSON.stringify({ model: LLM_MODEL, messages, stream: false }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`LLM error ${res.status}: ${detail.slice(0, 300)}`);
  }
  const json = await res.json();
  return String(json.choices?.[0]?.message?.content ?? "");
}

// ---------- Routes: conversations ----------

app.get("/api/health", async () => ({ ok: true }));

app.get("/api/conversations", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const snap = await conversations.orderBy("updatedAt", "desc").limit(50).get();
  return snap.docs.map((d) => ({
    id: d.id,
    title: d.data().title ?? "Untitled",
    updatedAt: tsToMillis(d.data().updatedAt),
  }));
});

app.get<{ Params: { id: string } }>(
  "/api/conversations/:id/messages",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const snap = await conversations
      .doc(request.params.id)
      .collection("messages")
      .orderBy("createdAt", "asc")
      .get();

    return snap.docs.map((d) => ({
      role: d.data().role,
      content: d.data().content,
      createdAt: tsToMillis(d.data().createdAt),
    }));
  }
);

app.delete<{ Params: { id: string } }>(
  "/api/conversations/:id",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    await db.recursiveDelete(conversations.doc(request.params.id));
    return { ok: true };
  }
);

// ---------- Routes: memory CRUD ----------

app.get("/api/memory", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const snap = await memory.orderBy("createdAt", "desc").limit(500).get();
  return snap.docs.map((d) => ({
    id: d.id,
    text: d.data().text ?? "",
    active: d.data().active !== false,
    createdAt: tsToMillis(d.data().createdAt),
    sourceConversationId: d.data().sourceConversationId ?? null,
  }));
});

app.post<{ Body: { text?: string } }>("/api/memory", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const text = request.body?.text?.trim();
  if (!text) return reply.code(400).send({ error: "text required" });

  const doc = await memory.add({
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

    const updates: Record<string, unknown> = {};
    if (typeof request.body?.text === "string" && request.body.text.trim()) {
      updates.text = request.body.text.trim();
    }
    if (typeof request.body?.active === "boolean") {
      updates.active = request.body.active;
    }
    if (Object.keys(updates).length === 0) {
      return reply.code(400).send({ error: "Nothing to update" });
    }

    await memory.doc(request.params.id).update(updates);
    return { ok: true };
  }
);

// ---------- Route: extraction ("Remember this conversation") ----------

const EXTRACTION_INSTRUCTIONS = `You maintain a long-term memory of durable facts about the user.

Given the conversation transcript and the list of EXISTING facts, respond with ONLY a JSON object, no markdown fences, in exactly this shape:
{"new_facts": ["..."], "deactivate_ids": ["..."]}

Rules:
- new_facts: durable facts about the user worth remembering across future conversations (preferences, projects, people, decisions, circumstances). Write each as one short standalone sentence about "the user".
- Do NOT include facts already covered by an existing fact.
- Do NOT include trivia, small talk, or one-off details with no future value.
- deactivate_ids: IDs of existing facts that this conversation shows are now false, outdated, or superseded.
- If nothing qualifies, return {"new_facts": [], "deactivate_ids": []}.`;

interface ExtractionResult {
  new_facts: string[];
  deactivate_ids: string[];
}

function parseExtraction(raw: string): ExtractionResult | null {
  // Tolerate accidental code fences despite instructions
  const cleaned = raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
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

app.post<{ Params: { id: string } }>(
  "/api/conversations/:id/remember",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    const conversationId = request.params.id;

    const msgSnap = await conversations
      .doc(conversationId)
      .collection("messages")
      .orderBy("createdAt", "asc")
      .get();

    if (msgSnap.empty) {
      return reply.code(400).send({ error: "Conversation has no messages" });
    }

    const transcript = msgSnap.docs
      .map((d) => `${d.data().role === "user" ? "User" : "Assistant"}: ${d.data().content}`)
      .join("\n");

    const existing = await loadActiveFacts();
    const existingBlock =
      existing.length > 0
        ? existing.map((f) => `[${f.id}] ${f.text}`).join("\n")
        : "(none)";

    const extractionMessages: ChatMessage[] = [
      { role: "system", content: EXTRACTION_INSTRUCTIONS },
      {
        role: "user",
        content: `EXISTING FACTS:\n${existingBlock}\n\nTRANSCRIPT:\n${transcript}`,
      },
    ];

    // One attempt + one retry on parse failure
    let result: ExtractionResult | null = null;
    for (let attempt = 0; attempt < 2 && !result; attempt++) {
      try {
        const raw = await llmComplete(extractionMessages);
        result = parseExtraction(raw);
        if (!result) request.log.warn({ attempt, raw: raw.slice(0, 200) }, "extraction parse failed");
      } catch (err) {
        request.log.error(err, "extraction call failed");
      }
    }

    if (!result) {
      return reply.code(502).send({ error: "Extraction failed after retry" });
    }

    // Apply: deactivate superseded facts, add new ones
    const validIds = new Set(existing.map((f) => f.id));
    const batch = db.batch();

    let deactivated = 0;
    for (const id of result.deactivate_ids) {
      if (!validIds.has(id)) continue; // never trust model-invented IDs
      batch.update(memory.doc(id), { active: false });
      deactivated++;
    }

    for (const text of result.new_facts) {
      batch.set(memory.doc(), {
        text: text.trim(),
        active: true,
        createdAt: FieldValue.serverTimestamp(),
        sourceConversationId: conversationId,
      });
    }

    await batch.commit();

    request.log.info(
      { user: user.email, added: result.new_facts.length, deactivated },
      "memory extraction applied"
    );

    return {
      added: result.new_facts,
      deactivated,
    };
  }
);

// ---------- Route: chat ----------

app.post<{ Body: ChatBody }>("/api/chat", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const body = request.body ?? {};
  const baseSystem =
    body.system ?? "You are a helpful personal assistant. Be concise and direct.";

  // Memory injection happens for every chat, both paths
  const facts = await loadActiveFacts();
  const systemPrompt = buildSystemPrompt(baseSystem, facts);

  let providerMessages: ChatMessage[];
  let conversationId: string | null = null;

  if (typeof body.message === "string" && body.message.trim()) {
    const userText = body.message.trim();

    if (body.conversationId) {
      conversationId = body.conversationId;
    } else {
      const doc = await conversations.add({
        title: userText.slice(0, 60),
        createdAt: FieldValue.serverTimestamp(),
        updatedAt: FieldValue.serverTimestamp(),
      });
      conversationId = doc.id;
    }

    const convRef = conversations.doc(conversationId);

    await convRef.collection("messages").add({
      role: "user",
      content: userText,
      createdAt: FieldValue.serverTimestamp(),
    });

    const history = await loadHistory(conversationId);
    providerMessages = [{ role: "system", content: systemPrompt }, ...history];
  } else if (Array.isArray(body.messages) && body.messages.length > 0) {
    providerMessages = [
      { role: "system", content: systemPrompt },
      ...body.messages.filter((m) => m.role === "user" || m.role === "assistant"),
    ];
  } else {
    return reply.code(400).send({ error: "Provide 'message' or 'messages'" });
  }

  const upstream = await fetch(`${LLM_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages: providerMessages,
      stream: true,
    }),
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    request.log.error({ status: upstream.status, detail }, "LLM provider error");
    return reply
      .code(502)
      .send({ error: "Model provider error", status: upstream.status });
  }

  reply.raw.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Expose-Headers": "X-Conversation-Id",
    ...(conversationId ? { "X-Conversation-Id": conversationId } : {}),
  });

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let lineBuffer = "";
  let assistantText = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      reply.raw.write(value);

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
          const delta: string | undefined = json.choices?.[0]?.delta?.content;
          if (delta) assistantText += delta;
        } catch {
          // ignore partial frames
        }
      }
    }
  } catch (err) {
    request.log.error(err, "stream interrupted");
  } finally {
    reply.raw.end();
  }

  if (conversationId && assistantText) {
    const convRef = conversations.doc(conversationId);
    await convRef.collection("messages").add({
      role: "assistant",
      content: assistantText,
      createdAt: FieldValue.serverTimestamp(),
    });
    await convRef.update({ updatedAt: FieldValue.serverTimestamp() });
  }

  request.log.info(
    { user: user.email, conversationId, facts: facts.length, chars: assistantText.length },
    "chat completed"
  );
});

// ---------- Start ----------

app.listen({ port: PORT, host: "0.0.0.0" }).catch((err) => {
  app.log.error(err);
  process.exit(1);
});
