1. **Implement Spectral Operators (Phase E)**
   - Create `src/shaders/mulIM.wgsl`: compute `i * m * a(m,l)`
   - Create `src/shaders/applyLaplacian.wgsl`: compute `lapEigs * a`
   - Create `src/shaders/invertLaplacian.wgsl`: compute `a / (-l(l+1))` and handle `l=0`
   - Create `src/shaders/filterSpectrum.wgsl`: apply `specFilter` to `a`
   - Create `tests/shaders/spectralOperators.test.ts` to test these shaders and ensure correctness against CPU reference
2. **Implement Initialization (Phase F)**
   - Create `src/shaders/initRandom.wgsl`: generate a random grid, apply analysis, scaling, and filtering
   - Create `tests/shaders/initRandom.test.ts`
3. **Pre-commit steps**
   - Call `pre_commit_instructions` tool to run verification and testing, ensuring no errors or omissions in `tests/shaders`
   - Update `task.md` checkboxes
4. **Submit changes**
   - Call the `submit` tool
