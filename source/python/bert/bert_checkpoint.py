from typing import Any

import glob
import re
import shutil
import os

def sort_chekpoints (args : Any, checkpoint_prefix : str = 'checkpoint', use_mtime : bool = False) :
	"""
	Doc
	"""

	ordering_and_checkpoint_path = list()
	glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))

	for path in glob_checkpoints :
		if use_mtime :
			ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
		else :
			regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)

			if regex_match and regex_match.groups() :
				ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

	checkpoints_sorted = sorted(ordering_and_checkpoint_path)
	checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

	return checkpoints_sorted

def rotate_checkpoints (args : Any, checkpoint_prefix : str = 'checkpoint', use_mtime : bool = False) :
	"""
	Doc
	"""

	if not args.save_total_limit :
		return
	if args.save_total_limit <= 0 :
		return

	checkpoints_sorted = sort_chekpoints(
		args              = args,
		checkpoint_prefix = checkpoint_prefix,
		use_mtime         = use_mtime
	)

	if len(checkpoints_sorted) <= args.save_total_limit :
		return

	number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
	checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

	for checkpoint in checkpoints_to_be_deleted :
		shutil.rmtree(checkpoint)
