import numpy as np

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

def StateElection(State, BidenPoll, TrumpPoll, MOE):
	BidenDelegates = 0
	TrumpDelegates = 0

	BidenResult = np.random.normal(BidenPoll, MOE/2)
	TrumpResult = TrumpPoll + (BidenPoll - BidenResult)


	if BidenResult > TrumpResult:
		BidenDelegates = EC[State]
	else:
		TrumpDelegates = EC[State]

	return BidenDelegates, TrumpDelegates

OV_B = 0
OV_T = 0

for i in range(0,len(States)):

	B, T = StateElection(States[i], .50, .50, .04)

	OV_B = OV_B + B
	OV_T = OV_T + T
	#print(States[i])

print(OV_B)
print(OV_T)