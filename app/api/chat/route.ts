import { GoogleGenerativeAIStream, StreamingTextResponse, LangChainStream, Message } from 'ai';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { AIMessage, HumanMessage } from 'langchain/schema';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const runtime = 'edge';

export async function POST(req: Request) {
  const { messages } = await req.json();
  const currentMessageContent = messages[messages.length - 1].content;

  const vectorSearch = await fetch("http://localhost:3000/api/vectorSearch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: currentMessageContent,
  })
  .then((res) => res.json())
  .then((data) => data.map((m: any) => m.pageContent));

  const TEMPLATE = `Answer the question as precise as possible using the provided context. If the answer is not contained in the context, say "answer not available in context" \n\n
  
  Context sections:
  ${JSON.stringify(vectorSearch)}

  Question: """
  ${currentMessageContent}
  """
  `;

  console.log(TEMPLATE);

  messages[messages.length -1].content = TEMPLATE;

  if(process.env.CHAT_MODEL == "GOOGLE"){
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "")
    const geminiStream = await genAI
      .getGenerativeModel({ model: 'gemini-pro' })
      .generateContentStream(
        {
          contents: (messages as Message[])
            .filter(m => m.role === 'user' || m.role === 'assistant')
            .map(m => ({
              role: m.role === 'user' ? 'user' : 'model',
              parts: [{ text: m.content }],
            })),
        }
      );
    const stream = GoogleGenerativeAIStream(geminiStream);
    return new StreamingTextResponse(stream);
  }
  // OpenAI
  const { stream, handlers } = LangChainStream();

  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
  });

  llm
    .call(
      (messages as Message[]).map(m =>
        m.role == 'user'
          ? new HumanMessage(m.content)
          : new AIMessage(m.content),
      ),
      {},
      [handlers],
    )
    .catch(console.error);

  return new StreamingTextResponse(stream);
}