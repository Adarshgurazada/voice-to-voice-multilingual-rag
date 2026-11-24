const $ = (id) => document.getElementById(id);
const form = $("document-form");
const files = $("pdf-files");
const connectButton = $("connect-button");
const muteButton = $("mute-button");
const languageSelect = $("answer-language");
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
let workingCard = null;  // the "checking your documents" progress card

// Follow the newest words, but leave the learner alone once they scroll up to reread something.
let followTranscript = true;
function pinTranscript() {
  const box = $("transcript");
  if (followTranscript) box.scrollTop = box.scrollHeight;
}
document.addEventListener("DOMContentLoaded", () => {
  const box = $("transcript");
  box.addEventListener("scroll", () => {
    followTranscript = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  });
});
let liveRow = null;  // the "as you speak" line, replaced by the final transcript

// The Realtime API produces audio and its transcript faster than real time, but the voice plays at
// normal speed. Text for each response is therefore revealed against the audio playback clock.
const REVEAL_LEAD = 0.25;  // seconds the text may run ahead of the voice
const audioSegments = new Map();  // response item id -> { start, seconds } on the audio clock
const revealing = new Set();  // assistant rows whose text is still being revealed
let audioItem = null;  // the item the incoming audio chunks belong to
let revealFrame = null;

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

function sendLanguage() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "set_language", language: languageSelect.value }));
  }
}

function message(text) { $("call-message").textContent = text; }

function visibleLength(row, now) {
  const text = row._text || "";
  const segment = audioSegments.get(row._itemId);
  if (!audioContext) return text.length;
  if (!segment) return performance.now() - row._created > 1500 ? text.length : 0;  // no audio to follow
  if (segment.seconds < 0.05) return 0;
  const rate = text.length / segment.seconds;  // characters per second of audio
  const end = Math.floor(Math.max(0, now - segment.start + REVEAL_LEAD) * rate);
  if (end >= text.length) return text.length;
  if (end <= 0) return 0;  // this response's audio has not started yet
  let word = end;  // never stop mid-word
  while (word < text.length && !/\s/.test(text[word])) word++;
  return word;
}

function paintRow(row, length) {
  const content = row.querySelector("p");
  const text = (row._text || "").slice(0, length);
  if (length > 0 && !row.isConnected) $("transcript").append(row);
  if (row._finished && row._citations?.length) renderCitations(row, text, row._citations);
  else content.textContent = text;
  pinTranscript();
}

function revealTick() {
  revealFrame = null;
  const now = audioContext ? audioContext.currentTime : 0;
  for (const row of [...revealing]) {
    const total = (row._text || "").length;
    const length = visibleLength(row, now);
    if (row._final && length >= total) {
      row._finished = true;
      revealing.delete(row);
    }
    paintRow(row, length);
  }
  if (revealing.size) revealFrame = requestAnimationFrame(revealTick);
}

function startReveal(row) {
  revealing.add(row);
  if (revealFrame === null) revealFrame = requestAnimationFrame(revealTick);
}

// The learner interrupted: keep only what was actually spoken.
function cutRevealingText() {
  const now = audioContext ? audioContext.currentTime : 0;
  for (const row of [...revealing]) {
    const total = (row._text || "").length;
    const length = Math.min(visibleLength(row, now), total);
    row._text = length < total ? `${row._text.slice(0, length).trimEnd()}\u2026` : row._text;
    row._final = row._finished = row._interrupted = true;
    revealing.delete(row);
    paintRow(row, row._text.length);
  }
}

function stopPlayback() {
  cutRevealingText();
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
  if (audioItem) {
    const segment = audioSegments.get(audioItem) || { start, seconds: 0 };
    segment.seconds += buffer.duration;
    audioSegments.set(audioItem, segment);
  }
  state("Speaking", true);
}

function escapeRegExp(text) { return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

// Words the server respelled from the documents' vocabulary are marked so the fix is visible.
function renderCorrected(content, text, corrections) {
  content.replaceChildren();
  const targets = [...new Set(corrections.map((fix) => fix.to))];
  const pattern = new RegExp(targets.map(escapeRegExp).join("|"), "g");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    content.append(document.createTextNode(text.slice(last, match.index)));
    const span = document.createElement("span");
    span.className = "corrected";
    span.textContent = match[0];
    const fix = corrections.find((item) => item.to === match[0]);
    span.title = `Heard \u201c${fix.from}\u201d, corrected to a term from your documents`;
    content.append(span);
    last = match.index + match[0].length;
  }
  content.append(document.createTextNode(text.slice(last)));
}

function showStatus(event) {
  const transcript = $("transcript");
  if (event.state === "start" || (!workingCard && event.state === "step")) {
    workingCard = document.createElement("details");
    workingCard.className = "working";
    workingCard.open = true;
    const summary = document.createElement("summary");
    const loader = document.createElement("span");
    loader.className = "loader";
    loader.setAttribute("aria-hidden", "true");
    loader.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
    const title = document.createElement("span");
    title.className = "working-title";
    title.textContent = event.query ? `Checking your documents for \u201c${event.query}\u201d` : "Checking your documents";
    summary.append(loader, title);
    const bar = document.createElement("div");
    bar.className = "working-bar";
    bar.setAttribute("aria-hidden", "true");
    const steps = document.createElement("ol");
    steps.className = "steps";
    workingCard.append(summary, bar, steps);
    transcript.append(workingCard);
  }
  if (!workingCard) return;
  const steps = workingCard.querySelector(".steps");
  if (event.state === "step") {
    for (const item of steps.querySelectorAll("li.active")) item.classList.replace("active", "done");
    const item = document.createElement("li");
    item.className = "active";
    item.textContent = event.text;
    steps.append(item);
  } else if (event.state === "end") {
    for (const item of steps.querySelectorAll("li.active")) item.classList.replace("active", "done");
    workingCard.classList.add("finished");
    workingCard.classList.toggle("empty", Boolean(event.abstained));
    workingCard.querySelector(".working-title").textContent = event.summary || "Search finished";
    workingCard.open = false;  // collapse; the learner can expand to see every step
    workingCard = null;
  }
  pinTranscript();
}

function showLive(text) {
  if (!liveRow) {
    liveRow = document.createElement("div");
    liveRow.className = "utterance user partial live";
    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = "You";
    liveRow.append(speaker, document.createElement("p"));
    $("transcript").append(liveRow);
  }
  liveRow.querySelector("p").textContent = text;
  pinTranscript();
}

function clearLive() {
  if (liveRow) { liveRow.remove(); liveRow = null; }
}

function addTranscript(event) {
  const key = `${event.speaker}:${event.item_id || "current"}`;
  let row = transcriptRows.get(key);
  if (!row && event.speaker === "user" && liveRow) {
    // Keep the words already on screen; the accurate transcript replaces them when it is final.
    row = liveRow;
    liveRow = null;
    row._adopted = true;
    transcriptRows.set(key, row);
  }
  if (row && row._adopted && !event.final) return;
  if (!row) {
    row = document.createElement("div");
    row.className = `utterance ${event.speaker} partial`;
    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = event.speaker === "user" ? "You" : "Companion";
    const content = document.createElement("p");
    row.append(speaker, content);
    if (event.speaker === "assistant") {
      row._itemId = event.item_id;
      row._created = performance.now();
      row._text = "";
    } else {
      $("transcript").append(row);
    }
    transcriptRows.set(key, row);
  }
  if (event.speaker === "assistant") {
    if (row._interrupted) return;  // the learner cut this answer off; late text is not shown
    // Held back and revealed in step with the voice (see revealTick).
    if (event.final) {
      row._text = event.text || row._text;
      row._sources = [...currentSources];
      row._final = true;
      row.classList.remove("partial");
    } else {
      row._text += event.delta || "";
    }
    startReveal(row);
    return;
  }
  const content = row.querySelector("p");
  if (event.final) {
    const finalText = event.text || content.textContent;
    if (event.corrections?.length) renderCorrected(content, finalText, event.corrections);
    else content.textContent = finalText;
    if (event.speaker === "assistant") row._sources = [...currentSources];
    row.classList.remove("partial", "live");
  } else {
    content.textContent += event.delta || "";
  }
  pinTranscript();
}

const ROLE_NAMES = {
  answer: "Answer", definition: "Definition", step: "Step", reason: "Why", example: "Example", caution: "Caution",
};

function openPage(card, article) {
  const image = article.querySelector("img.page");
  const note = article.querySelector(".page-note");
  const button = article.querySelector("button.show");
  for (const other of document.querySelectorAll("#evidence-cards article.open")) {
    if (other !== article) {
      other.classList.remove("open");
      other.querySelector("img.page").hidden = true;
      other.querySelector("button.show").textContent = "Show on page";
    }
  }
  const opening = !article.classList.contains("open");
  article.classList.toggle("open", opening);
  image.hidden = !opening;
  button.textContent = opening ? "Hide page" : "Show on page";
  if (!opening) return;
  article.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (!image.src) {
    note.textContent = card.located === null ? "Reading the scanned page to find the quote\u2026" : "Highlighting\u2026";
    image.onload = () => {
      note.textContent = card.located === false
        ? "The exact words could not be located on the page, so the whole page is shown."
        : "The quote is highlighted.";
    };
    image.onerror = () => { image.hidden = true; note.textContent = "The page image is unavailable."; };
    image.src = card.url;
  }
}

function showEvidence(event) {
  const container = $("evidence-cards");
  container.replaceChildren();
  document.querySelector(".evidence").scrollTop = 0;
  $("evidence-hint").hidden = !(event.cards?.length || event.reading);
  for (const card of event.cards || []) {
    const article = document.createElement("article");
    article.className = `card role-${card.role}`;
    const header = document.createElement("header");
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = ROLE_NAMES[card.role] || card.role;
    const where = document.createElement("span");
    where.className = "where";
    where.textContent = card.label;
    header.append(chip, where);
    const claim = document.createElement("p");
    claim.className = "claim";
    claim.textContent = card.claim;
    const quote = document.createElement("blockquote");
    quote.className = "quote";
    quote.textContent = `\u201c${card.quote}\u201d`;
    const footer = document.createElement("footer");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "show";
    button.textContent = "Show on page";
    button.onclick = () => openPage(card, article);
    const pageNote = document.createElement("span");
    pageNote.className = "page-note";
    footer.append(button, pageNote);
    const image = document.createElement("img");
    image.className = "page";
    image.alt = `${card.label} with the quote highlighted`;
    image.hidden = true;
    article.append(header, claim, quote, footer, image);
    container.append(article);
  }
  if (event.reading) {
    // The companion's own plain-language reading: shown apart from the quoted evidence and labelled as such.
    const article = document.createElement("article");
    article.className = "card reading";
    const header = document.createElement("header");
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = "Companion\u2019s reading";
    const where = document.createElement("span");
    where.className = "where";
    where.textContent = "Not quoted from the document. Checked: it adds no fact the verified claims lack.";
    header.append(chip, where);
    const text = document.createElement("p");
    text.className = "claim";
    text.textContent = event.reading;
    article.append(header, text);
    container.append(article);
  }
  const first = container.querySelector("article.card:not(.reading)");
  if (first) first.querySelector("button.show").click();
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
    sendLanguage();
  } else if (event.type === "preferences") {
    // The model can change the language by voice ("answer in Hindi"); keep the dropdown in sync.
    if ([...languageSelect.options].some((option) => option.value === event.language)) {
      languageSelect.value = event.language;
    }
  } else if (event.type === "audio_item") {
    audioItem = event.item_id;
  } else if (event.type === "evidence") {
    showEvidence(event);
  } else if (event.type === "status") {
    showStatus(event);
  } else if (event.type === "live_transcript") {
    showLive(event.text || "");
  } else if (event.type === "speech_started") {
    followTranscript = true;  // a new turn: follow it again
    clearLive();
    stopPlayback();
    state("Listening", true);
  } else if (event.type === "speech_stopped") {
    state("Thinking", true);
  } else if (event.type === "transcript_discard") {
    // The transcription model only echoed its hint; nothing was said, so drop any row it started.
    clearLive();
    const echoed = transcriptRows.get(`user:${event.item_id || "current"}`);
    if (echoed) { echoed.remove(); transcriptRows.delete(`user:${event.item_id || "current"}`); }
  } else if (event.type === "transcript") {
    addTranscript(event);
  } else if (event.type === "sources") {
    showSources(event.sources || []);
  } else if (event.type === "citations") {
    const row = transcriptRows.get(`assistant:${event.item_id || "current"}`);
    if (row && event.spans?.length) {
      row._citations = event.spans;  // applied as soon as the text has finished being revealed
      if (row._finished) renderCitations(row, row._text, event.spans);
    }
  } else if (event.type === "metrics") {
    const first = event.metrics.time_to_first_audio_ms;
    const answer = event.metrics.time_to_answer_audio_ms;
    $("voice-metrics").textContent = first == null ? "No response audio"
      : answer == null ? `First audio ${first} ms`
      : `First audio ${first} ms · Answer ${answer} ms`;
  } else if (event.type === "turn_done") {
    clearLive();
    workingCard = null;
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
  clearLive();
  $("evidence-cards").replaceChildren();
  revealing.clear();
  audioSegments.clear();
  audioItem = null;
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
  $("evidence-cards").replaceChildren();
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

languageSelect.addEventListener("change", sendLanguage);
connectButton.addEventListener("click", () => stream ? disconnect() : connect());
muteButton.addEventListener("click", () => {
  muted = !muted;
  muteButton.textContent = muted ? "Unmute mic" : "Mute mic";
  muteButton.setAttribute("aria-pressed", String(muted));
  message(muted ? "Microphone muted" : "Microphone live");
});
window.addEventListener("beforeunload", () => { if (socket) socket.close(); });
