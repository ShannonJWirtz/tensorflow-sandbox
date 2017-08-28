from tfsource import *
from datasource import *
#probgens

#logsitic
def logistic(x, l = 0, u = 1, m = 0, s = 1):
	return l + (u-l)/(1 + exp(s*(m-x)))

#bp in this format: [bp1, bp2, bp3, ..., bpn]
#p in this format [value_before_bp1, 
#either value between bp1 and bp2 or list of start and end values, 
#same for bp2 and bp3, value after bpn]
'''
def pwl(x, bp, p):
	for i in range(len(bp)):
		if np.where(x < bp[i],
			if isinstance(p[i], (int, float)):
				return p[i]
			return p[i][0] + (p[i][1] - p[i][0])*(y - bp[i-1])/(bp[i] - bp[i-1])
	return p[-1]


pwl = np.vectorize(pwl_proto)
'''

def pwl(x, bp, p):
	if np.array(x).ndim == 0:
		x = np.array([x])
	x = x.astype(float)
	condlist = []
	funclist = []

	for i in range(len(bp)):
		if i == 0:
			#condlist.append([x < bp[i]])
			condlist.append(x < bp[i])
		elif i < len(bp) - 1:
			#condlist.append([np.logical_and(x >= bp[i-1], x < bp[i])])
			condlist.append(np.logical_and(x >= bp[i-1], x < bp[i]))
		else:
			#condlist.append([x >= bp[-1]])
			condlist.append(x >= bp[-1])

	for i in range(len(p)):
		if isinstance(p[i], (int, float)):
			funclist.append(float(p[i]))
		else:
			funclist.append(lambda x,i=i: p[i][0] + (p[i][1] - p[i][0])*(x - bp[i-1])/(bp[i] - bp[i-1]))

	return np.piecewise(x, np.array(condlist), np.array(funclist))
	#return [condlist, funclist]






#examples:
def pwl1(x):
	return pwl(x, [-4, 15, 100], [-3, [-3, 0], 18, 3])
