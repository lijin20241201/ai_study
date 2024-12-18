# 后处理序列
def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    eos_pos = seq.tolist().index(eos_idx)
    seq = [
        idx for idx in seq[:eos_pos + 1] # 切片到eos_pos
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq