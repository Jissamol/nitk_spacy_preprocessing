import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config(); // Load .env file

const app = express();
app.use(cors());
app.use(bodyParser.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.post("/api/generate", async (req, res) => {
  const { prompt, genre, tone, length } = req.body;

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const finalPrompt = `Write a ${tone} ${genre} story of about ${length} words. 
    Start with: ${prompt}`;

    const result = await model.generateContent(finalPrompt);

    res.json({ story: result.response.text() });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(4000, () => {
  console.log("✅ Backend running on http://localhost:4000");
});
