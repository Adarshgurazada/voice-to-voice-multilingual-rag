const $ = (id) => document.getElementById(id);
const form = $("document-form");
const files = $("pdf-files");
const connectButton = $("connect-button");
const muteButton = $("mute-button");
let sessionId = null;
let socket = null;
let stream = null;
let audioContext = null;
let inputNode = null;
let workletNode = null;
let muted = false;
let reconnecting = false;
let nextAudioTime = 0;
const playing = new Set();
const transcriptRows = new Map();
const sourceButtons = new Map();
let currentSources = [];

function renderCitations(row, answer, spans) {
  const content = row.querySelector("p");
  content.replaceChildren();
  let start = 0;
  for (const span of spans) {
    if (span.start < start || span.end > answer.length || span.end <= span.start) continue;
    content.append(document.createTextNode(answer.slice(start, span.end)));
    const source = (row._sources || []).find((item) => item.id === span.source_id);
    if (source) {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "citation-link";
      link.textContent = `[${span.source_id}]`;
      link.title = `${source.label}: ${span.evidence_quote}`;
      link.onclick = () => {
        showSources(row._sources);
        sourceButtons.get(span.source_id)?.click();
      };
      content.append(link);
    } else {
      content.append(document.createTextNode(` [${span.source_id}]`));
    }
    start = span.end;
  }
  content.append(document.createTextNode(answer.slice(start)));
}

function state(label, online = false) {
  $("call-state").textContent = label;
  $("state-dot").classList.toggle("online", online);
  $("turn-state").textContent = label;
}

function message(text) { $("call-message").textContent = text; }

function stopPlayback() {
  for (const source of playing) {
    try { source.stop(); } catch (_) { /* A completed buffer may already be stopped. */ }
  }
  playing.clear();
  nextAudioTime = audioContext ? audioContext.currentTime : 0;
}

function playPcm(data) {
  if (!audioContext || data.byteLength < 2) return;
  const pcm = new Int16Array(data);
  const buffer = audioContext.createBuffer(1, pcm.length, 24000);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const start = Math.max(audioContext.currentTime + 0.04, nextAudioTime);
  source.start(start);
  nextAudioTime = start + buffer.duration;
  playing.add(source);
  source.onended = () => playing.delete(source);
  state("Speaking", true);
}

function addTranscript(event) {
  const key = `${event.speaker}:${event.item_id || "current"}`;
  let row = transcriptRows.get(key);
  if (!row) {
    row = document.createElement("div");
    row.className = `utterance ${event.speaker} partial`;
    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = event.speaker === "user" ? "You" : "Companion";
    const content = document.createElement("p");
    row.append(speaker, content);
    $("transcript").append(row);
    transcriptRows.set(key, row);
  }
  const content = row.querySelector("p");
  if (event.final) {
    const finalText = event.text || content.textContent;
    content.textContent = finalText;
    if (event.speaker === "assistant") row._sources = [...currentSources];
    row.classList.remove("partial");
  } else {
    content.textContent += event.delta || "";
  }
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

function showSources(sources) {
  currentSources = [...sources];
  const container = $("sources");
  container.replaceChildren();
  sourceButtons.clear();
  const image = $("page-image");
  image.hidden = true;
  image.removeAttribute("src");
  for (const source of sources) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = source.id ? `${source.id} · ${source.label}` : source.label;
    button.dataset.label = source.label;
    button.title = source.excerpt || source.label;
    if (source.id) sourceButtons.set(source.id, button);
    button.onclick = () => {
      for (const other of container.querySelectorAll("button")) other.classList.remove("active");
      button.classList.add("active");
      image.src = source.url;
      image.hidden = false;
      $("source-excerpt").textContent = source.excerpt || "";
    };
    container.append(button);
  }
  if (container.firstChild) container.firstChild.click();
  else $("source-excerpt").textContent = "";
}

function handleServer(event) {
  if (event.type === "ready") {
    state("Listening", true);
    message("Microphone live");
  } else if (event.type === "speech_started") {
    stopPlayback();
    state("Listening", true);
  } else if (event.type === "speech_stopped") {
    state("Thinking", true);
  } else if (event.type === "transcript") {
    addTranscript(event);
  } else if (event.type === "sources") {
    showSources(event.sources || []);
  } else if (event.type === "citations") {
    const row = transcriptRows.get(`assistant:${event.item_id || "current"}`);
    if (row && event.spans?.length) renderCitations(row, row.querySelector("p").textContent, event.spans);
  } else if (event.type === "metrics") {
    const value = event.metrics.time_to_first_audio_ms;
    $("voice-metrics").textContent = value == null ? "No response audio" : `First audio ${value} ms`;
  } else if (event.type === "turn_done") {
    state("Listening", true);
    transcriptRows.delete("user:current");
    transcriptRows.delete("assistant:current");
  } else if (event.type === "reconnect") {
    reconnecting = true;
    state("Reconnecting", false);
    socket.close();
  } else if (event.type === "error") {
    message(event.message || "Voice connection failed");
    state("Error", false);
  }
}

function openSocket() {
  return new Promise((resolve, reject) => {
    let settled = false;
    const ready = () => { if (!settled) { settled = true; resolve(); } };
    const failed = () => { if (!settled) { settled = true; reject(new Error("Voice connection failed")); } };
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${scheme}//${location.host}/ws/${sessionId}`);
    socket.binaryType = "arraybuffer";
    socket.onmessage = (messageEvent) => {
      if (typeof messageEvent.data === "string") {
        const event = JSON.parse(messageEvent.data);
        handleServer(event);
        if (event.type === "ready") ready();
      } else {
        playPcm(messageEvent.data);
      }
    };
    socket.onerror = failed;
    socket.onclose = async () => {
      failed();
      if (reconnecting) {
        reconnecting = false;
        try { await openSocket(); } catch (error) { message(error.message); await disconnect(); }
      } else if (stream) {
        message("Connection closed");
        await disconnect();
      }
    };
  });
}

async function connect() {
  if (!sessionId) return;
  connectButton.disabled = true;
  state("Connecting");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    audioContext = new AudioContext({ sampleRate: 24000 });
    await audioContext.audioWorklet.addModule("/assets/mic-worklet.js");
    inputNode = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, "mic-capture");
    workletNode.port.onmessage = (event) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        if (muted) socket.send(new Int16Array(2400).buffer);
        else socket.send(event.data);
      }
    };
    inputNode.connect(workletNode);
    workletNode.connect(audioContext.destination);
    await audioContext.resume();
    await openSocket();
    connectButton.textContent = "Disconnect";
    connectButton.classList.add("disconnect");
    muteButton.disabled = false;
  } catch (error) {
    message(error.message || "Could not open microphone");
    await disconnect();
  } finally {
    connectButton.disabled = false;
  }
}

async function disconnect() {
  reconnecting = false;
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
  stopPlayback();
  if (inputNode) inputNode.disconnect();
  if (workletNode) workletNode.disconnect();
  if (stream) for (const track of stream.getTracks()) track.stop();
  if (audioContext) await audioContext.close();
  inputNode = workletNode = stream = audioContext = null;
  nextAudioTime = 0;
  muted = false;
  muteButton.disabled = true;
  muteButton.textContent = "Mute mic";
  muteButton.setAttribute("aria-pressed", "false");
  connectButton.textContent = "Connect";
  connectButton.classList.remove("disconnect");
  state("Offline");
}

async function pollIndex(id) {
  while (sessionId === id) {
    const response = await fetch(`/api/sessions/${id}`);
    const data = await response.json();
    $("progress-bar").value = data.progress;
    $("progress-text").textContent = `${data.progress}% - ${data.detail}`;
    if (data.status === "ready") {
      $("document-name").textContent = data.documents.join(", ");
      connectButton.disabled = false;
      message("Documents ready");
      return;
    }
    if (data.status === "error") throw new Error(data.detail);
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!files.files.length) return;
  if (files.files.length > 5 || [...files.files].some((file) => file.size > 25 * 1024 * 1024)) {
    message("Choose up to five PDFs, each at most 25 MB.");
    return;
  }
  if (stream) await disconnect();
  transcriptRows.clear();
  $("transcript").replaceChildren();
  showSources([]);
  $("document-name").textContent = "No documents loaded";
  $("index-button").disabled = true;
  $("index-progress").hidden = false;
  $("progress-bar").value = 0;
  $("progress-text").textContent = "Preparing documents";
  connectButton.disabled = true;
  try {
    const response = await fetch("/api/sessions", { method: "POST", body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not upload PDFs");
    sessionId = data.session_id;
    await pollIndex(sessionId);
  } catch (error) {
    message(error.message || "Indexing failed");
    $("progress-text").textContent = error.message || "Indexing failed";
  } finally {
    $("index-button").disabled = false;
  }
});

connectButton.addEventListener("click", () => stream ? disconnect() : connect());
muteButton.addEventListener("click", () => {
  muted = !muted;
  muteButton.textContent = muted ? "Unmute mic" : "Mute mic";
  muteButton.setAttribute("aria-pressed", String(muted));
  message(muted ? "Microphone muted" : "Microphone live");
});
window.addEventListener("beforeunload", () => { if (socket) socket.close(); });
