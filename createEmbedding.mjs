import { promises as fsp } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { MongoClient } from "mongodb";
import "dotenv/config";

// 連結MongoDB
const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const dbName = "docs";
const collectionName = "embeddings";
const collection = client.db(dbName).collection(collectionName);

// 讀取文字檔案
const docs_dir = "_assets/fcc_docs";
const fileNames = await fsp.readdir(docs_dir);
console.log(fileNames);
for (const fileName of fileNames) {
  const document = await fsp.readFile(`${docs_dir}/${fileName}`, "utf8");
  console.log(`Vectorizing ${fileName}`);

  const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const output = await splitter.createDocuments([document]);
  // 選擇模型
  const embedModel = process.env.CHAT_MODEL === "GOOGLE"
    ? new GoogleGenerativeAIEmbeddings()
    : new OpenAIEmbeddings();
  // 寫入MongoDB   
  await MongoDBAtlasVectorSearch.fromDocuments(
    output,
    embedModel,
    {
      collection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding",
    }
  );
}

console.log("Done: Closing Connection");
await client.close();