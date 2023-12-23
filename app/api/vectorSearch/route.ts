import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import mongoClientPromise from '@/app/lib/mongodb';

export async function POST(req: Request) {
  const client = await mongoClientPromise;
  const dbName = "docs";
  const collectionName = "embeddings";
  const collection = client.db(dbName).collection(collectionName);
  
  const question = await req.text();

  const aiprops =  {
    collection,
    indexName: "default",
    textKey: "text", 
    embeddingKey: "embedding",
  }
  const vectorStore = process.env.CHAT_MODEL === "GOOGLE" 
    ? new MongoDBAtlasVectorSearch(
        new GoogleGenerativeAIEmbeddings({stripNewLines: true}), aiprops)   
    : new MongoDBAtlasVectorSearch(
        new OpenAIEmbeddings({modelName: 'text-embedding-ada-002', stripNewLines: true}), aiprops);

  const retriever = vectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: {
      fetchK: 20,
      lambda: 0.1,
    },
  });
  
  const retrieverOutput = await retriever.getRelevantDocuments(question);
  
  return Response.json(retrieverOutput);
}