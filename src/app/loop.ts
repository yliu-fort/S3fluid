export class AppLoop {
    private animationId: number = 0;
    public isPaused: boolean = false;
    private lastTime: number = 0;

    constructor(
        private stepFn: () => void,
        private renderFn: () => void,
        private getStepsPerFrame: () => number
    ) {}

    start() {
        this.lastTime = performance.now();
        this.animationId = requestAnimationFrame((time) => this.tick(time));
    }

    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
    }

    private tick(time: number) {
        if (!this.isPaused) {
            const steps = this.getStepsPerFrame();
            for (let i = 0; i < steps; i++) {
                this.stepFn();
            }
        }

        this.renderFn();

        this.lastTime = time;
        this.animationId = requestAnimationFrame((t) => this.tick(t));
    }
}
