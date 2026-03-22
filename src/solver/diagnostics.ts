export class Diagnostics {
    public energyHistory: number[] = [];
    public historyLength: number = 1000;

    constructor() {}

    recordEnergy(energy: number) {
        this.energyHistory.push(energy);
        if (this.energyHistory.length > this.historyLength) {
            this.energyHistory.shift();
        }
    }

    reset() {
        this.energyHistory = [];
    }

    getLatestEnergy(): number | null {
        if (this.energyHistory.length > 0) {
            return this.energyHistory[this.energyHistory.length - 1];
        }
        return null;
    }
}
