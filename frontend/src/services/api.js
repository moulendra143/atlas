import axios from "axios";

const API = import.meta.env.VITE_API_BASE_URL || (window.location.origin + "/api");

export const api = {
  getState: () => axios.get(`${API}/state`),
  reset: (preset) => axios.post(`${API}/reset`, { preset }),
  leaderboard: () => axios.get(`${API}/leaderboard`),
  replay: (episodeId) => axios.get(`${API}/replay/${episodeId}`),
  investorReport: (episodeId) => `${API}/investor-report/${episodeId}`,
  pause: () => axios.post(`${API}/pause`),
  resume: () => axios.post(`${API}/resume`),
  speed: (val) => axios.post(`${API}/speed?val=${val}`)
};
