const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS = import.meta.env.VITE_WS_BASE_URL || `${protocol}//${window.location.host}/ws`;

export function connectWS(onEvent, onStatusChange) {
  let ws = null;
  let reconnectTimer = null;
  let closedByUser = false;
  let attempt = 0;

  const connect = () => {
    if (onStatusChange) onStatusChange(attempt === 0 ? "Connecting..." : `Retrying... (Attempt ${attempt})`);
    ws = new WebSocket(WS);
    
    ws.onopen = () => {
      attempt = 0;
      if (onStatusChange) onStatusChange("Online");
    };

    ws.onmessage = (msg) => onEvent(JSON.parse(msg.data));
    
    ws.onclose = () => {
      if (closedByUser) return;
      if (onStatusChange) onStatusChange("Disconnected");
      const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
      attempt++;
      reconnectTimer = window.setTimeout(connect, delay);
    };
    
    ws.onerror = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  };

  connect();

  return {
    close() {
      closedByUser = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      if (ws) {
        ws.close();
      }
    },
  };
}
