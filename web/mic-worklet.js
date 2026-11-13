// Collects 16 kHz mono mic audio into 80 ms Int16 frames (1280 samples) and posts them.
class MicFrames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Int16Array(1280);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let sum = 0;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      sum += s * s;
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.buf.length) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Int16Array(1280);
        this.n = 0;
      }
    }
    this.port.postMessage({ rms: Math.sqrt(sum / ch.length) });
    return true;
  }
}
registerProcessor("mic-frames", MicFrames);
