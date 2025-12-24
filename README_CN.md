# Manufacturing AI Code & Report Assistant 🔧



## 功能特性

✅ **AI代碼生成** - 自然語言 → Python分析腳本  
✅ **自動報告** - Markdown / HTML 一鍵生成  
✅ **製造業專用** - 針對SPC、良率、缺陷分析優化  
✅ **多模型支持** - Mistral、ChatGPT、本地Ollama  
✅ **Docker部署** - 一鍵上雲 (AWS/GCP)  

## 快速開始

### 本地運行
\`\`\`bash
git clone https://github.com/your-repo.git
cd manufacturing-ai-assistant
pip install -r requirements.txt
streamlit run app.py
\`\`\`

### Docker運行
\`\`\`bash
docker-compose up
# 訪問 http://localhost:8501
\`\`\`

## 使用範例

1. 上傳製程CSV檔案
2. 輸入需求："分析良率Top3異常因子"
3. AI生成Python代碼 + 報告
4. 下載Markdown/HTML報告

## 效能指標

| 指標 | 數值 |
|------|------|
| 代碼生成準確率 | >85% |
| 平均生成時間 | <10秒 |
| 支持檔案大小 | <100MB |
| 可用模型 | 3+ |

## 技術棧

- **前端**: Streamlit
- **LLM**: HuggingFace/OpenAI/Ollama
- **資料處理**: Pandas/NumPy/Scikit-learn
- **部署**: Docker + AWS (可選)

## 專案結構

\`\`\`
src/
├── config.py          # 全局設定
├── data_processor.py  # 資料處理
├── code_generator.py  # LLM調用 (核心)
├── report_generator.py # 報告生成
└── utils.py

notebooks/
├── 01_data_exploration.ipynb
├── 02_finetuning_codellama.ipynb
└── 03_demo_walkthrough.ipynb

app.py                 # Streamlit主應用
\`\`\`

## 路線圖

- [ ] 支援中文模型微調 (ChatGLM、Qwen)
- [ ] WebUI優化 + 暗色主題
- [ ] 生成代碼的單元測試自動產生
- [ ] 實時代碼執行(沙箱)
- [ ] API版本 (FastAPI)
- [ ] AWS Lambda 無伺服器部署

## 貢獻

歡迎PR! 詳見 [CONTRIBUTING.md](CONTRIBUTING.md)

## 許可證

MIT License

## 聯絡

📧 [your-email@example.com](mailto:your-email@example.com)  
🔗 [個人網站](https://your-portfolio.com)  
💼 [LinkedIn](https://linkedin.com/in/your-profile)

***

**⭐ 如果覺得有用，請給個Star!**
