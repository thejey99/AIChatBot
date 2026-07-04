import Fastify from "fastify";
import cors from "@fastify/cors";
import { readFileSync } from "node:fs";
import { initializeApp, cert } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore, FieldValue, Timestamp } from "firebase-admin/firestore";

// ---------- Configuration (Render: Secret Files + environment variables) ----------

const PORT = Number(process.env.PORT ?? 8080);

// Comma-separated allowlist. THE lock on the whole app.
const ALLOWED_EMAILS = new Set(
  (process.env.ALLOWED_EMAILS ?? "jryan@charlestownehotels.com,john99ran@gmail.com")
    .split(",")
    .map((e) => e.trim().toLowerCase())
);

const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";

// LLM provider — OpenAI-compatible. Defaults target Gemini's free tier.
const LLM_BASE_URL =
  process.env.LLM_BASE_URL ??
  "https://generativelanguage.googleapis.com/v1beta/openai";
const LLM_MODEL = process.env.LLM_MODEL ?? "gemini-2.5-flash";
const LLM_API_KEY = process.env.LLM_API_KEY ?? "";

// How many recent messages to send as context per turn.
// The compaction layer (memory phase) will make this smarter.
const HISTORY_LIMIT = 30;

// Firebase Admin credentials: Secret File preferred, env var fallback.
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

// ---------- Auth: verify Firebase ID token + enforce allowlist ----------

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

// Small helper so every route guards identically
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
  message?: string;          // new persistent contract
  conversationId?: string;   // omit to start a new conversation
  messages?: ChatMessage[];  // legacy stateless fallback
  system?: string;
}

// ---------- Firestore helpers ----------

const conversations = db.collection("conversations");

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

// ---------- Routes ----------

app.get("/api/health", async () => ({ ok: true }));

// List conversations, newest activity first
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

// Full message history for one conversation
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

// Delete a conversation and all its messages
app.delete<{ Params: { id: string } }>(
  "/api/conversations/:id",
  async (request, reply) => {
    const user = await guard(request, reply);
    if (!user) return;

    await db.recursiveDelete(conversations.doc(request.params.id));
    return { ok: true };
  }
);

// Chat: persistent by default, legacy stateless fallback preserved
app.post<{ Body: ChatBody }>("/api/chat", async (request, reply) => {
  const user = await guard(request, reply);
  if (!user) return;

  const body = request.body ?? {};
  const systemPrompt =
    body.system ?? "You are a helpful personal assistant. Be concise and direct.";

  let providerMessages: ChatMessage[];
  let conversationId: string | null = null;

  if (typeof body.message === "string" && body.message.trim()) {
    // ----- Persistent path -----
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

    // Persist the user message before calling the model
    await convRef.collection("messages").add({
      role: "user",
      content: userText,
      createdAt: FieldValue.serverTimestamp(),
    });

    const history = await loadHistory(conversationId);
    providerMessages = [{ role: "system", content: systemPrompt }, ...history];
  } else if (Array.isArray(body.messages) && body.messages.length > 0) {
    // ----- Legacy stateless path (old frontend keeps working) -----
    providerMessages = [
      { role: "system", content: systemPrompt },
      ...body.messages.filter((m) => m.role === "user" || m.role === "assistant"),
    ];
  } else {
    return reply.code(400).send({ error: "Provide 'message' or 'messages'" });
  }

  // ----- Call the provider, streaming -----
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

  // Forward SSE to the client while accumulating the assistant's full
  // reply server-side so it can be persisted when the stream ends.
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

      reply.raw.write(value); // forward raw bytes to the client

      // Parse the same bytes to accumulate the reply for persistence
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

  // Persist the assistant reply (persistent path only)
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
    { user: user.email, conversationId, chars: assistantText.length },
    "chat completed"
  );
});

// ---------- Start ----------

app.listen({ port: PORT, host: "0.0.0.0" }).catch((err) => {
  app.log.error(err);
  process.exit(1);
});
