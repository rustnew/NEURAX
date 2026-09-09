/**
 * A live view onto a run that exists on disk.
 *
 * This replaces the simulated clock the Training workspace was built against.
 * The views above it are unchanged — that was the point of building them
 * against a clock in the first place: what gets replaced is where the steps
 * come from, not how they are drawn.
 *
 * Three things it does that a naive poll would not:
 *
 *  - **It asks only for what it does not have.** Each poll sends back the
 *    byte offset the last one returned, so the service seeks past the history
 *    this studio already holds. Without it, watching a 100 000-step run means
 *    re-transferring and re-parsing 100 000 steps every second, and the cost
 *    grows exactly as the run gets more interesting.
 *
 *  - **It stops polling a run that has stopped.** A finished run does not
 *    change, and a studio left open on one should not keep asking. Polling
 *    resumes on its own if the run is restarted, because the last poll before
 *    stopping recorded a terminal status, and starting a run replaces the id.
 *
 *  - **It survives the service going away.** A failed request is not an
 *    error state — the run is on disk and the service can come back — so the
 *    hook keeps what it had and retries, rather than blanking the view.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { getTrainingRun } from '@/services/neuraxApi.ts';
import type { Checkpoint, ModelBuiltEvent, RunState, TrainingStep } from '@/types/runtime.ts';

/** How often to ask, while a run is moving. Fast enough that a loss curve
 *  grows visibly, slow enough that a thousand-step-per-second run is not
 *  answering a request per step. */
const POLL_MS = 1000;

export interface LiveRun {
  state: RunState | null;
  model: ModelBuiltEvent | null;
  steps: TrainingStep[];
  checkpoints: Checkpoint[];
  /** True while the run is still moving — the studio uses it to decide
   *  whether Pause and Stop mean anything. */
  isLive: boolean;
  /** The service did not answer the last poll. The run is not lost; nothing
   *  is listening to it right now. */
  disconnected: boolean;
}

const EMPTY: LiveRun = {
  state: null,
  model: null,
  steps: [],
  checkpoints: [],
  isLive: false,
  disconnected: false,
};

function isTerminal(state: RunState | null): boolean {
  return (
    state?.status === 'finished' || state?.status === 'failed' || state?.status === 'interrupted'
  );
}

export function useTrainingRun(runId: string | null): LiveRun & { refresh: () => void } {
  const [run, setRun] = useState<LiveRun>(EMPTY);
  // Held in a ref, not in state: it changes on every poll and nothing renders
  // from it, so putting it in state would re-render the whole workspace once a
  // second for a number no one sees.
  const offset = useRef(0);
  const stopped = useRef(false);

  const poll = useCallback(async () => {
    if (!runId) return;
    const snapshot = await getTrainingRun(runId, offset.current);

    if (!snapshot) {
      // Keep everything already drawn. A dropped request means the service is
      // not answering, which says nothing about the run — it is a directory,
      // and it is still there.
      setRun((prev) => ({ ...prev, disconnected: true }));
      return;
    }

    offset.current = snapshot.nextOffset;
    stopped.current = isTerminal(snapshot.state);

    setRun((prev) => ({
      state: snapshot.state,
      model: snapshot.model ?? prev.model,
      // Appended, not replaced: the response holds only what is new.
      steps: snapshot.steps.length > 0 ? [...prev.steps, ...snapshot.steps] : prev.steps,
      checkpoints: snapshot.checkpoints,
      isLive: snapshot.state.status === 'running' || snapshot.state.status === 'paused',
      disconnected: false,
    }));
  }, [runId]);

  useEffect(() => {
    if (!runId) {
      setRun(EMPTY);
      offset.current = 0;
      stopped.current = false;
      return;
    }

    // A new run id is a different run: everything held about the previous one
    // is wrong for this one, including how far into its step file we had read.
    setRun(EMPTY);
    offset.current = 0;
    stopped.current = false;

    let cancelled = false;
    void poll();

    const timer = setInterval(() => {
      if (cancelled) return;
      // A run that has finished, failed or been interrupted does not change.
      // Asking again would be a request per second, forever, for a file that
      // is not being written.
      if (stopped.current) return;
      void poll();
    }, POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId, poll]);

  /** Ask now, rather than waiting for the next tick.
   *
   *  Used after a control command: pausing and then waiting a second to see
   *  the button change makes the control feel broken, and re-enables polling
   *  for a run that had stopped and has just been restarted. */
  const refresh = useCallback(() => {
    stopped.current = false;
    void poll();
  }, [poll]);

  return { ...run, refresh };
}
