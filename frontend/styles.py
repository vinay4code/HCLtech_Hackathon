def get_custom_css() -> str:
    """Returns modern, high-contrast, professional Enterprise SaaS styling for Streamlit."""
    return """
    <style>
    /* Global Typography & Colors */
    :root {
        --primary-navy: #0f172a;
        --sidebar-bg: #1e293b;
        --accent-blue: #2563eb;
        --accent-teal: #0d9488;
        --success-green: #16a34a;
        --warning-amber: #d97706;
        --danger-red: #dc2626;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        color: #94a3b8;
    }

    /* Metric Card Component */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-primary { background: #dbeafe; color: #1e40af; }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-danger { background: #fee2e2; color: #991b1b; }
    .badge-neutral { background: #f1f5f9; color: #475569; }

    /* 3-Panel Approval Column Styling */
    .panel-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        height: 100%;
    }
    .panel-box-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }

    /* Quality Warning Banner */
    .warning-banner {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: #92400e;
    }

    /* Section Header */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """
