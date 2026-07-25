import os
import pdb
import shlex
import sys
from .python_wer_evaluation import wer_calculation


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    # Use the running interpreter, not a bare "python": on macOS/venv setups that
    # name often does not exist, mergectmstm.py silently never runs, and clips the
    # model predicted nothing for stay missing from the CTM -- which then raises a
    # KeyError in wer_calculation and reports the whole split as 100% WER.
    # Quote it: the interpreter path may contain spaces.
    os.system(f"{shlex.quote(sys.executable)} {evaluate_dir}/mergectmstm.py "
              f"{prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
        if triplet:
            wer_calculation(
                f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
