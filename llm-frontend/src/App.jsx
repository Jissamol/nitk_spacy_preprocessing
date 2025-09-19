import { useState } from "react";
import "./App.css";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [genre, setGenre] = useState("fantasy");
  const [tone, setTone] = useState("adventurous");
  const [length, setLength] = useState(300);
  const [story, setStory] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleGenerate(e) {
    e.preventDefault();
    setLoading(true);
    setStory("");

    try {
      const res = await fetch("http://localhost:4000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, genre, tone, length }),
      });
      

      const data = await res.json();
      setStory(data.story);
    } catch (err) {
      console.error(err);
      setStory("⚠️ Error generating story");
    }
    setLoading(false);
  }

  return (
    <div className="app-root">
      <div className="container">
        <header className="app-header">
          <h1>📖 AI Story Generator</h1>
          <p className="subtitle">Turn an idea into a story — pick genre, tone, and length.</p>
        </header>

        <form className="form" onSubmit={handleGenerate}>
          <label className="label">Story idea</label>
          <textarea
            className="textarea"
            rows={4}
            placeholder="Enter your story idea..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <div className="row">
            <div className="field">
              <label className="label">Genre</label>
              <select className="select" value={genre} onChange={(e) => setGenre(e.target.value)}>
                <option value="fantasy">Fantasy</option>
                <option value="sci-fi">Sci-Fi</option>
                <option value="romance">Romance</option>
                <option value="horror">Horror</option>
                <option value="mystery">Mystery</option>
              </select>
            </div>

            <div className="field">
              <label className="label">Tone</label>
              <select className="select" value={tone} onChange={(e) => setTone(e.target.value)}>
                <option value="adventurous">Adventurous</option>
                <option value="serious">Serious</option>
                <option value="funny">Funny</option>
                <option value="dark">Dark</option>
                <option value="romantic">Romantic</option>
              </select>
            </div>

            <div className="field small">
              <label className="label">Length (words)</label>
              <input
                className="number"
                type="number"
                value={length}
                onChange={(e) => setLength(e.target.value)}
                min={100}
                max={1000}
              />
            </div>
          </div>

          <div className="actions">
            <button className="btn" type="submit" disabled={loading}>
              {loading ? "Generating..." : "Generate Story"}
            </button>
            <button
              className="btn btn-ghost"
              type="button"
              onClick={() => {
                setPrompt("");
                setStory("");
              }}
            >
              Clear
            </button>
          </div>
        </form>

        <section className="output-section">
          <h2 className="output-title">✨ Generated Story:</h2>
          <div className="output">
            {story ? <pre className="story-text">{story}</pre> : <div className="placeholder">Your story will appear here...</div>}
          </div>
        </section>

        <footer className="footer">
          <small>Tip: Short prompts + strong genre/tone give faster, more focused stories.</small>
        </footer>
      </div>
    </div>
  );
}
