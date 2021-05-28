def calculate_mean(metric):
	metric_sum = sum(i for i in metric)
	return metric_sum / len(metric)
