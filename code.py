import numpy as np
import pandas as pd
# Recursive Bayesian Estimation
# Kalman Filter
# https://www-oxfordhandbooks-com.stanford.idm.oclc.org/view/10.1093/oxfordhb/9780190213299.001.0001/oxfordhb-9780190213299-e-28
# https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/01-g-h-filter.ipynb
# https://www.johndcook.com/blog/2012/10/29/product-of-normal-pdfs/
# https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html

# Import polling data
df = pd.read_csv('data.csv', index_col = 0)
df.insert(df.shape[1], "PA_Biden", '')
df.insert(df.shape[1], "PA_Trump", '')
df.insert(df.shape[1], "PA_SD", '')


# Dictionary of Electoral College Votes
EC = {	'AK':3, 'AL':9, 'AR':6, 'AZ':11, 'CA':55, 'CO': 9, 'CT':7, 'DE':3, 'FL':29, 'GA':16, \
		'HI':4, 'IA':6, 'ID':4, 'IL':20, 'IN':11, 'KS':6, 'KY':8, 'LA':8, 'MA':11, 'MD':10, \
		'ME':4, 'MI':16, 'MN':10, 'MO':10, 'MS':6, 'MT':3, 'NC':15, 'ND':3, 'NE':5, 'NH':4, \
		'NJ':14, 'NM':5, 'NV':6, 'NY':29, 'OH':18, 'OK':7, 'OR':7, 'PA':20, 'RI':4, 'SC':9, \
		'SD':3, 'TN':11, 'TX':38, 'UT':6, 'VA':13, 'VT':3, 'WA':12, 'WI':10, 'WV':5, 'WY':3, \
		'DC':3}


# List of States
States = (	'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', \
			'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', \
			'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', \
			'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', \
			'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'DC')


def StateElection():
	BidenDelegates = 0
	TrumpDelegates = 0

	for i in range(0, len(States)):
		State = States[i]

		BidenResult = np.random.normal(df.PA_Biden[State], df.PA_SD[State])
		TrumpResult = np.random.normal(df.PA_Trump[State], df.PA_SD[State])

		if BidenResult > TrumpResult:
			BidenDelegates = BidenDelegates + EC[State]
		else:
			TrumpDelegates = TrumpDelegates + EC[State]

	return BidenDelegates, TrumpDelegates


# Description on bayesion updating of gausian / normal PDFs
def BayesianUpdate(PriorMU, PriorSD, LikelihoodMU, LikelihoodSD):
	PosteriorMU = (((PriorSD**2)*(LikelihoodMU)) + ((LikelihoodSD**2)*(PriorMU))) \
		/ ((PriorSD**2) + (LikelihoodSD**2))

	PosteriorSD = (((PriorSD**2)*(LikelihoodSD**2)) \
					/ ((PriorSD**2) + (LikelihoodSD**2)))**(1/2)

	return PosteriorMU, PosteriorSD



def StateAggregator():
	for i in range(0, len(States)):
		State = States[i]

		# Biden - aggregate the first and second poles
		# Divide margin of error (95% confidence interval) for standard deviation
		BidenMU, BidenSD = BayesianUpdate(df.P1_Biden[State], df.P1_MOE[State]/1.96, \
										  df.P2_Biden[State], df.P2_MOE[State]/1.96)
		# Aggregate these with the third pole
		BidenMU, BidenSD = BayesianUpdate(BidenMU, BidenSD, \
										  df.P3_Biden[State], df.P3_MOE[State]/1.96)
		
		# Trump - aggregate the first and second poles
		# Divide margin of error (95% confidence interval) for standard deviation
		TrumpMU, TrumpSD = BayesianUpdate(df.P1_Trump[State], df.P1_MOE[State]/1.96, \
										  df.P2_Trump[State], df.P2_MOE[State]/1.96)
		# Aggregate these with the third pole
		TrumpMU, TrumpSD = BayesianUpdate(TrumpMU, TrumpSD, \
										  df.P3_Trump[State], df.P3_MOE[State]/1.96)

		# Update the global dataframe
		df.PA_Biden[State] = BidenMU
		df.PA_Trump[State] = TrumpMU

		# The standard deviation of the aggregated poll is equal for Biden and Trump
		df.PA_SD[State] = BidenSD


StateAggregator()

b, t = StateElection()
print(b)
print(t)