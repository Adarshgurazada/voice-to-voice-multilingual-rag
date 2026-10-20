class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = [];
    this.cursor = 0;
    this.packet = [];
    this.step = sampleRate / 24000;
  }

  process(inputs, outputs) {
    for (const channel of outputs[0]) channel.fill(0);
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (const sample of input) this.samples.push(sample);
    while (this.cursor + 1 < this.samples.length) {
      const left = Math.floor(this.cursor);
      const fraction = this.cursor - left;
      const value = this.samples[left] * (1 - fraction) + this.samples[left + 1] * fraction;
      this.packet.push(Math.round(Math.max(-1, Math.min(1, value)) * 32767));
      this.cursor += this.step;
      if (this.packet.length === 2400) {
        const data = Int16Array.from(this.packet);
        this.port.postMessage(data.buffer, [data.buffer]);
        this.packet = [];
      }
    }
    const consumed = Math.floor(this.cursor);
    if (consumed > 0) {
      this.samples.splice(0, consumed);
      this.cursor -= consumed;
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
