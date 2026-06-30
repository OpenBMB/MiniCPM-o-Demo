export class BoardState {
  constructor({ maxCards = 10 } = {}) {
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
