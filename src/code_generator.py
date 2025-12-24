import logging
from typing import Optional, Dict, List
import requests
import json
from src.config import (
    LLM_MODEL, LLM_API_KEY, USE_LOCAL_MODEL, 
    LOCAL_API_URL, MAX_TOKENS, TEMPERATURE, TOP_P
)

logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    調用LLM (HuggingFace、OpenAI 或 本地 Ollama/vLLM) 生成Python代碼與分析。
    """
    
    def __init__(self, model_name: str = LLM_MODEL, api_key: str = LLM_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.use_local = USE_LOCAL_MODEL
        self.local_url = LOCAL_API_URL
        
        # 系統提示詞 (製造領域特定)
        self.system_prompt = """You are an expert Manufacturing Engineer AI Assistant specializing in:
- Statistical Process Control (SPC) and Quality Analysis.
- Python Data Analysis (Pandas/Numpy) and Streamlit Visualization.
- Semiconductor manufacturing insights (Yield, OEE, Defect Root Cause).

### DATA CONTEXT:
- **Available Columns (Schema):** {sample_dataframe_schema}
- **Data Preview:** {data_summary}

### CORE RULES:
1. **Dynamic Column Mapping (CRITICAL):** - DO NOT assume column names (e.g., 'good_count'). 
   - ALWAYS use a "fuzzy search" approach to find columns. 
   - Example: For "Yield", search for ['yield_pct', '良率', 'Yield_Rate', 'Pass_Rate'].
   - If a required column is missing, explain it and try to use alternatives from the schema.

2. **Streamlit Display Rules:**
   - Visuals: ALWAYS use `fig, ax = plt.subplots()` followed by `st.pyplot(fig)`. NEVER use `plt.show()`.
   - Tables: Use `st.dataframe()` for results.

3. **Output Language:** - Code comments: **Traditional Chinese (繁體中文)**.
   - AI Insights: **Traditional Chinese (繁體中文)**.

### TASKS:
1. Generate clean, production-ready Python code within 
tags.
2. Ensure the code includes proper error handling for missing columns.
3. Provide actionable manufacturing insights in Markdown format after the code block.

### CODE SNIPPET TEMPLATE:
```python
# 1. 欄位自動對齊 (Mapping)
def find_col(possible_names, df_columns):
    for n in possible_names:
        if n in df_columns: return n
    return None

target_col = find_col(['yield_pct', '良率', 'Yield'], df.columns)
# 2. 進行運算與顯示...

"""
    
    def _call_openai(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return resp.choices[0].message.content
    

    def generate_code(self,
                      user_request: str,
                      data_summary: str,
                      sample_dataframe_schema: Dict) -> Dict[str, str]:
        """
        根據用戶自然語言需求生成Python代碼。

        Args:
            user_request: 用戶需求 (e.g., "分析本週良率Top3異常因子")
            data_summary: 資料摘要
            sample_dataframe_schema: DataFrame欄位資訊

        Returns:
            {'code': str, 'insights': str, 'explanation': str}
        """

        prompt = self._build_prompt(user_request, data_summary, sample_dataframe_schema)

        # response = self._call_openai(prompt)

        try:
            if self.use_local:
                response = self._call_local_llm(prompt)
            else:
                response = self._call_huggingface_or_openai(prompt)

            # 解析回應
            code, insights = self._parse_response(response)

            logger.info(f"Generated code for: {user_request[:50]}...")

            return {
                'code': code,
                'insights': insights,
                'explanation': self._generate_explanation(user_request, code),
            }

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                'code': '# Error generating code\nprint("Error: Check logs")',
                'insights': f'Failed to generate insights: {str(e)}',
                'explanation': '',
            }

    def _build_prompt(self, user_request: str, data_summary: str, schema: Dict) -> str:
        """組合prompt (few-shot learning)"""

        few_shot_example = """
Example:
User Request: "分析良率趨勢並找出Top3異常原因"
Data: yield, temperature, pressure, defect_count columns

Response:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

良率趨勢分析
df['yield_pct'] = (df['good_count'] / df['total_count'] * 100).round(2)
trend = df.groupby('date')['yield_pct'].mean()

異常檢測 (Top3最低良率的原因)
low_yield = df[df['yield_pct'] < df['yield_pct'].quantile(0.25)]
top3_factors = low_yield[['temperature', 'pressure', 'defect_count']].describe()

print("Top 3 Low Yield Factors:")
print(top3_factors)

視覺化
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
trend.plot(ax=ax, title='Yield Trend', marker='o')
low_yield[['temperature', 'pressure']].boxplot(ax=ax)​
plt.tight_layout()
plt.show()

Insights:
- Yield下降主要與溫度波動>3°C相關
- 壓力異常導致defect rate提升50%
"""

        prompt = f"""{self.system_prompt}

{few_shot_example}

---

Current Request:
User: {user_request}

Data Info:
{data_summary}

Schema: {json.dumps(schema, ensure_ascii=False)}

Please generate Python code to address the user request.
Output format:

Then provide insights in markdown format."""

        return prompt

    def _call_huggingface_or_openai(self, prompt: str) -> str:
        """呼叫HuggingFace或OpenAI API"""

        if "gpt" in self.model_name.lower():
            return self._call_openai(prompt)
        else:
            return self._call_huggingface(prompt)

    def _call_huggingface(self, prompt: str) -> str:
        """HuggingFace Inference API (需API token)"""

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list):
            return result[0].get('generated_text', '')
        return result.get('generated_text', '')

    def _call_openai(self, prompt: str) -> str:
        """OpenAI API (ChatGPT)"""

        try:
            import openai
            openai.api_key = self.api_key

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            return response['choices'][0]['message']['content']

        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise

    def _call_local_llm(self, prompt: str) -> str:
        """呼叫本地 LLM (Ollama chat API)"""

        url = f"{self.local_url}/api/chat"  # 期望是 http://ollama:11434/api/chat
        payload = {
            "model": self.model_name,  # qwen2.5:1.5b
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_predict": MAX_TOKENS,
            },
        }

        logger.info(f"Calling local LLM at {url}, model={self.model_name}")

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


    def _parse_response(self, response: str) -> tuple:
        """從LLM回應中解析代碼與insights"""
        import re

        code = ""
        insights = ""

        # 抓 `````` 的 code block
        pattern = r"```python\s*(.*?)\s*```"
        code_match = re.search(pattern, response, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        # 把 code block 後面的文字當作 insights
        if "```" in response:
            remaining = response.split("```")[-1]
        else:
            remaining = response
        insights = remaining.strip()

        return code, insights





    def _generate_explanation(self, request: str, code: str) -> str:
        """生成代碼說明"""

        return f"根據需求 '{request}' 生成的Python分析腳本，包含資料清理、分析與視覺化步驟。"