const BACKEND_URL = "https://your-backend.onrender.com";

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

function appendMessage(text, role) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

async function sendMessage() {
  const question = userInput.value.trim();
  if (!question) return;

  userInput.value = "";
  sendBtn.disabled = true;

  appendMessage(question, "user");
  const loader = appendMessage("Thinking...", "loading");

  try {
    const res = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    loader.remove();
    appendMessage(data.answer, "bot");
  } catch (err) {
    loader.remove();
    appendMessage("Error: Could not reach the server.", "bot");
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}
