import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useStore = create(
  persist(
    (set) => ({
      state: null,
      mode: 'startup',
      done: false,
      episodeId: null,
      events: [],
      decisions: [],
      rewards: [],
      history: [],
      moraleHistory: [],
      leaderboard: [],
      currentDay: 1,
      currentPhase: 'morning',

      setState: (state) => set({ state }),
      setMode: (mode) => set({ mode }),
      setDone: (done) => set({ done }),
      setEpisodeId: (episodeId) => set({ episodeId }),
      setEvents: (updater) => set((s) => ({ events: typeof updater === 'function' ? updater(s.events) : updater })),
      setDecisions: (updater) => set((s) => ({ decisions: typeof updater === 'function' ? updater(s.decisions) : updater })),
      setRewards: (updater) => set((s) => ({ rewards: typeof updater === 'function' ? updater(s.rewards) : updater })),
      setHistory: (updater) => set((s) => ({ history: typeof updater === 'function' ? updater(s.history) : updater })),
      setMoraleHistory: (updater) => set((s) => ({ moraleHistory: typeof updater === 'function' ? updater(s.moraleHistory) : updater })),
      setLeaderboard: (updater) => set((s) => ({ leaderboard: typeof updater === 'function' ? updater(s.leaderboard) : updater })),
      setCurrentDay: (currentDay) => set({ currentDay }),
      setCurrentPhase: (currentPhase) => set({ currentPhase }),

      resetSimulation: (newState, newEpisodeId) => set({
        state: newState,
        episodeId: newEpisodeId,
        currentDay: 1,
        currentPhase: 'morning',
        events: [],
        decisions: [],
        rewards: [],
        history: [],
        moraleHistory: [{ name: 'S1', mood: newState.employee_morale }],
        done: false,
      }),

      resetForReplay: () => set({
        events: [],
        decisions: [],
        rewards: [],
        history: [],
        moraleHistory: [],
        done: false,
      })
    }),
    {
      name: 'atlas_state',
      partialize: (state) => ({
        // We persist everything except leaderboard and active mode since they can be re-fetched/re-set easily
        state: state.state,
        mode: state.mode,
        done: state.done,
        episodeId: state.episodeId,
        events: state.events,
        decisions: state.decisions,
        rewards: state.rewards,
        history: state.history,
        moraleHistory: state.moraleHistory,
        currentDay: state.currentDay,
        currentPhase: state.currentPhase,
      })
    }
  )
);
