from tools.result_loader import format_table

def test_table():
    t=format_table([{'scene':'truck','dataset':'Tanks and Temples','method':'baseline','psnr':1,'ssim':2,'lpips':3,'fps':4}])
    assert 'SEGS Quick Evaluation' in t and 'truck' in t
