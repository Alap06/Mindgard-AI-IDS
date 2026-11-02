"""
Mindgard — Autonomous LLM SOC Analyst Agent
Generates an emergency incident response playbook using Gemini.
"""
from __future__ import annotations

import os
import json
import time
from typing import Any
import requests

class SOCAgent:
    def __init__(self, api_key: str | None = None):
        # Default to the OpenRouter key from environment
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.is_mock = not bool(self.api_key)
    def generate_playbook(self, alert_data: dict[str, Any], shap_features: list[dict]) -> str:
        """
        Generate a Markdown playbook based on the threat signature.
        """
        prompt = self._build_prompt(alert_data, shap_features)

        if self.is_mock:
            return self._mock_response(alert_data)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8500",
                        "X-Title": "Mindgard IDS"
                    },
                    json={
                        "model": ["openrouter/free", "google/gemma-3-12b-it:free", "google/gemma-3-4b-it:free"][attempt % 3],
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=15, # Stop infinite hanging!
                )
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"].get("content")
                if not content:
                    return "**LLM returned an empty response.** Falling back to generic playbook...\n\n" + self._mock_response(alert_data)

                return content
            except Exception as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"**LLM Generation Error:** {e}\n\nFalling back to generic playbook...\n\n" + self._mock_response(alert_data)


    def _build_prompt(self, alert_data: dict[str, Any], shap_features: list[dict]) -> str:
        shap_str = "\n".join([f"- {f['feature']}: {f['shap_value']:.4f} (Raw value: {f['feature_value']})" for f in shap_features])

        prompt = f"""
You are an elite Tier-3 Cybersecurity Incident Responder evaluating an automated alert from a state-of-the-art Hybrid IDS (XGBoost + Isolation Forest).

ALERT CONTEXT:
{json.dumps(alert_data, indent=2)}

EXPLAINABILITY (SHAP Top Features):
{shap_str}

Formulate a highly professional, detailed, and tactical Incident Response Playbook in Markdown. Make it look like a real enterprise SOC platform output.
Include:
1. 🚨 **Threat Assessment & MITRE ATT&CK Mapping:** Detailed breakdown of what the SHAP features imply about the attack vector.
2. 🛡️ **Immediate Mitigation (Linux/CLI):** Robust command-line actions (iptables, fail2ban, ufw, etc.) to contain the threat.
3. 🔎 **Deep Investigation Steps:** What logs to check (`/var/log/auth.log`, pcap analysis, etc.) and what threat hunting to perform next.
4. 📈 **System Hardening:** Long-term recommendations to prevent recurrence.

Do not limit word count. Be brilliant, highly technical, and precise.
"""
        return prompt

    def _mock_response(self, alert_data: dict[str, Any]) -> str:
        ip = alert_data.get('IP', 'Unknown')
        pred = alert_data.get('Prediction', 'normal')

        if pred == "normal":
            action = "monitor"
            bash_cmd = f"echo 'Monitoring {ip}... No action needed.'"
        elif pred == "DoS":
            action = "block + rate limit"
            bash_cmd = f"iptables -A INPUT -s {ip} -j DROP\niptables -A INPUT -p tcp --syn -m limit --limit 50/s -j ACCEPT"
        elif pred == "Probe":
            action = "log + monitor escalation"
            bash_cmd = f"fail2ban-client set sshd banip {ip}"
        elif pred in ["R2L", "U2R"]:
            action = "isolate host + audit logs"
            bash_cmd = f"systemctl stop sshd\niptables -A INPUT -s {ip} -j DROP\n# Audit logs immediately"
        else:
            action = "quarantine + alert SOC"
            bash_cmd = f"iptables -A INPUT -s {ip} -j DROP\n# Quarantine zero-day ({pred})"

        return f"""
### 🚨 Automated SOC Playbook
> **Note:** Simulated response via Rule Engine because `$GEMINI_API_KEY` is not set.

**1. Threat Assessment:**
System executed behavior mapping. Identified `{pred}` signature from `{ip}`.

**2. Attack Class Mapping & Decision Action:**
- Policy Engine matched behavior pattern to action: **{action.upper()}**

**3. Immediate Mitigation Commands (Linux):**
```bash
# Automated Enforcement based on action ({action})
{bash_cmd}
```

**4. Investigation Steps:**
- Check logs (/var/log/auth.log, /var/log/syslog, network.log)
- Verify external threat intel sources (AlienVault OTX) for `{ip}`
"""
