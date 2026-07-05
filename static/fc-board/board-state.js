// Board FIFO state.
// 默认 maxCards = 6 = 2 x 3 grid（与训练数据 board 同期对象上限对齐；超出按 FIFO 挤出最早 card）。
export class BoardState {
  constructor({ maxCards = 6 } = {}) {
    this.maxCards = maxCards;
    this.cards = [];
  }

  upsert(card) {
    const idx = this.cards.findIndex((item) => item.card_id === card.card_id);
    if (idx >= 0) {
      this.cards[idx] = { ...this.cards[idx], ...card };
    } else {
      this.cards.push(card);
    }
    while (this.cards.length > this.maxCards) {
      this.cards.shift();
    }
    return this.cards;
  }
}
