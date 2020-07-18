def get_views(verbosity):
	total_views = ['default', 'thresh', 'mask', 'curv', 'angle', 'offset']
	if verbosity == 0:
		views = [total_views[0]]

	elif verbosity == 1:
		views = total_views[:3]

	elif verbosity == 2:
		views = total_views

	else:
		views = []
	return views