import Fastify from "fastify";
import cors from "@fastify/cors";
import { initializeApp, cert } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";

// ---------- Configuration (all via Render environment variables) ----------

const PORT = Number(process.env.PORT ?? 8080);

// Comma-separated allowlist. THE lock on the whole app.
const ALLOWED_EMAILS = new Set(
  (process.env.ALLOWED_EMAILS ?? "jryan@charlestownehotels.com,john99ran@gmail.com")
    .split(",")
    .map((e) => e.trim().toLowerCase())
);

// Frontend origin for CORS (your Render static site URL once it exists)
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";

// LLM provider — OpenAI-compatible. Defaults target Gemini's free tier.
// Swap provider later by changing these three env vars only.
const LLM_BASE_URL =
  process.env.LLM_BASE_URL ??
  "https://generativelanguage.googleapis.com/v1beta/openai";
const LLM_MODEL = process.env.LLM_MODEL ?? "gemini-2.5-flash";
const LLM_API_KEY = process.env.LLM_API_KEY ?? "";

// Firebase Admin — paste the full service-account JSON into this env var.
const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT ?? "{}");
initializeApp({ credential: cert(serviceAccount) });

// ---------- Server ----------

const app = Fastify({ logger: true });
await app.register(cors, { origin: ALLOWED_ORIGIN });

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

// ---------- Routes ----------

app.get("/api/health", async () => ({ ok: true }));

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatBody {
  messages: ChatMessage[];
  system?: string;
}

app.post<{ Body: ChatBody }>("/api/chat", async (request, reply) => {
  let user: AuthedUser;
  try {
    user = await requireAllowedUser(request.headers.authorization);
  } catch (err: any) {
    return reply.code(err.statusCode ?? 401).send({ error: err.message });
  }

  const { messages, system } = request.body ?? {};
  if (!Array.isArray(messages) || messages.length === 0) {
    return reply.code(400).send({ error: "messages array required" });
  }

  // Context assembly. The memory layer will slot in here in the next phase:
  // injected facts + rolling summary get prepended to the system prompt.
  const systemPrompt =
    system ??
    "You are a helpful personal assistant. Be concise and direct.";

  const providerMessages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    ...messages.filter((m) => m.role === "user" || m.role === "assistant"),
  ];

  // Call the provider with streaming enabled
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

  // Stream Server-Sent Events back to the client.
  // We forward the provider's OpenAI-format SSE stream as-is; the client
  // parses `data:` lines and extracts choices[0].delta.content.
  reply.raw.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  });

  const reader = upstream.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      reply.raw.write(value);
    }
  } catch (err) {
    request.log.error(err, "stream interrupted");
  } finally {
    reply.raw.end();
  }

  request.log.info({ user: user.email }, "chat completed");
});

// ---------- Start ----------

app.listen({ port: PORT, host: "0.0.0.0" }).catch((err) => {
  app.log.error(err);
  process.exit(1);
});
