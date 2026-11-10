"""Entry point: ``python run_evaluation.py``. The logic lives in evaluate_script.py.

This file is not named evaluate.py on purpose: a module called ``evaluate`` in this
directory would shadow the Hugging Face ``evaluate`` package used for ROUGE.
"""

from evaluate_script import main

if __name__ == "__main__":
    main()
