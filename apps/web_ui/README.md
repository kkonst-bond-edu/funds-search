# Web UI - Tab-Based Navigation

The Web UI provides a clean, modular interface for interacting with the Autonomous Vacancy Hunter system through a tab-based navigation structure.

## Architecture

The UI is organized into **4 main tabs**, each serving a distinct purpose:

### Tab 1: 💬 AI Recruiter
**Primary landing page** - Conversational job search interface.

- **Chat Interface**: Natural language interaction with the AI recruiter
- **Initial Greeting**: Russian greeting message on first load
- **API Endpoint**: Uses `POST /api/v1/vacancies/chat`
- **Features**:
  - Chat history stored in `st.session_state.chat_messages`
  - AI-generated summaries of search results
  - Vacancy cards displayed directly in chat flow
  - Error handling with friendly Russian messages

### Tab 2: 🔍 Manual Search
**Filter-based search** - Traditional search with explicit filters.

- **Filters**: Role, Industry, Skills, Location, Company Stage, Remote option
- **Search Button**: "Search Database" button
- **API Endpoint**: Uses `POST /api/v1/vacancies/search`
- **Features**:
  - All filters contained within the tab (not in sidebar)
  - Results displayed in grid format
  - Independent from chat history

### Tab 3: 📄 CV Analysis
**CV processing and matching** - Upload and analyze candidate resumes.

- **Sub-tabs**:
  - **Upload CV**: PDF upload and processing
  - **Process Vacancy**: Vacancy description processing
  - **Find Matches**: Candidate-vacancy matching
- **Features**:
  - PDF file uploader
  - Progress indicators for processing
  - Match results with AI reasoning
  - Summary statistics

### Tab 4: 🛠️ Diagnostics
**System health monitoring** - Comprehensive service diagnostics.

- **Features**:
  - Full system health check
  - Service status indicators
  - Latency metrics
  - Error diagnostics
  - Refresh functionality

## Sidebar

The sidebar is **clean and minimal**, containing only:

- **⚙️ Configuration**: API and CV Processor URLs
- **🏥 Service Health**: Status indicators for:
  - API Service
  - CV Processor
  - Embedding Service
- **🔄 Refresh Health Status**: Button to refresh health checks

**No filters or functional buttons** are in the sidebar - all functionality is contained within the appropriate tabs.

## Session State Management

- `chat_messages`: Stores chat history for AI Recruiter tab
- `interview_complete`: Tracks interview completion status
- `diagnostics_result`: Stores system diagnostics results
- `diagnostics_running`: Tracks diagnostics execution state

## Error Handling

Each tab handles errors gracefully:
- **AI Recruiter**: Friendly Russian error messages in chat
- **Manual Search**: Error messages with helpful guidance
- **CV Analysis**: Progress indicators and error states
- **Diagnostics**: Detailed error information with service-specific guidance

## Testing

To test the UI locally:

```bash
streamlit run apps/web_ui/app.py
```

**Verification Checklist**:
1. ✅ Sidebar is clean (only Configuration and Service Health)
2. ✅ Chat works in Tab 1 (AI Recruiter)
3. ✅ Manual filters work in Tab 2 (Manual Search) without affecting chat
4. ✅ CV Processor works in Tab 3 (CV Analysis)
5. ✅ Diagnostics work in Tab 4 (Diagnostics)

## Environment Variables

Required environment variables:
- `BACKEND_API_URL`: Backend API endpoint (default: `http://localhost:8000`)
- `CV_PROCESSOR_URL`: CV Processor service URL (default: `http://cv-processor:8001`)
- `EMBEDDING_SERVICE_URL`: Embedding service URL (default: `http://embedding-service:8001`)


