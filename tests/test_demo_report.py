from tools.report_generator import generate_report

def test_report_headless(tmp_path):
    p=generate_report([{'scene':'truck','dataset':'Tanks','method':'baseline'}], {}, {}, tmp_path, open_browser=False)
    assert p.exists() and 'SEGS Quick Evaluation' in p.read_text()
