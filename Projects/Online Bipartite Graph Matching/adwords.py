#UnityID- rkolhe

import sys
import csv
import pandas as pd
import random
import numpy as np

#used to analyse the input accordingly and save the csv files to dictionaries
def main():

	algo=''

	if sys.argv[1] == 'greedy':
		algo='greedy'
	elif sys.argv[1] == 'msvv':
		algo='msvv'
	elif sys.argv[1] == 'balance':
		algo='balance'
	else:
		print('invalid input')
	
	#extract all queries from the textfile
	with open('queries.txt') as f:
		queries = f.readlines()
	queries = [x.strip() for x in queries]

	#store the bidder data from the csv file
	bidder_data = pd.read_csv('bidder_dataset.csv')


	#store the budget of each advertiser in a dictionary
	#store the bid value of each keyword in the dictionary
	# Note: adv_budget = {advertiser: budget}
	#       keyword_adv =  {keyword1: [(advertiser1, bid_value1), (advertiser2, bid_value2)],
	#                        keyword2: [....], keyw...}  

	adv_budget={}
	keyword_adv={}

	for i in range(len(bidder_data)):
		advertiser = bidder_data.iloc[i]['Advertiser']
		bid_value = bidder_data.iloc[i]['Bid Value']
		keyword = bidder_data.iloc[i]['Keyword']
		budget = bidder_data.iloc[i]['Budget']

		if advertiser not in adv_budget:
			adv_budget[advertiser] = budget

		if keyword not in keyword_adv:
			keyword_adv[keyword] = [(advertiser, bid_value)]
		else: 
			keyword_adv[keyword].append((advertiser, bid_value))

	#calculates the optimal value used to calculate ratio
	optimum_budget = sum(adv_budget.values())

	revenue_ratio(queries, adv_budget, keyword_adv, optimum_budget, algo)

#Function used to compute the greedy algorithm
#returns the revenue(float) for all queries
def greedy(queries, new_budget, keyword_adv):
	revenue = 0.0
	all_spent = True
	#checks max_bid_value of all bidders for that query_advertiser
	for query in queries:

		#checks if all the neighbours have any budget left 
		for adv_bid in keyword_adv[query]:
			if new_budget[adv_bid[0]] >= adv_bid[1]:
				all_spent = False	 
		
		#if any neighbour has some budget left , it then enters the following loop
		if(all_spent == False):
			
			max_bid = -1 
			max_bidder_id = -1
			
			#checks bids for all advertisers for every query and selects the max bid
			for adv_bid in keyword_adv[query]:
				query_advertiser = adv_bid[0]
				advertiser_bid = adv_bid[1]
				advertiser_budget = new_budget[query_advertiser]
				#checks if budget of bidder for that particular query is more than the submitted bid
				if advertiser_budget >= advertiser_bid:
					#checks if the bid from that bidder is max bid
					if advertiser_bid > max_bid:	
						max_bidder_id = query_advertiser
						max_bid = advertiser_bid
					# if 2 bidder's have same bid then select the one with the least id
					if advertiser_bid == max_bid:
						if query_advertiser < max_bidder_id:
							max_bidder_id = query_advertiser
			#add the bid_value to the revenue and subtract bid value from bidder's budget
			#do this only if there is any bidder available with sufficient budget
			#if any of the above condition fails, the max_bid, and max_bidder wont get updated 
			if max_bidder_id != -1 and max_bid != -1: 
				revenue+= max_bid
				new_budget[max_bidder_id] -= max_bid

	return revenue

#Function used to compute the msvv algorithm
#returns the revenue(float) for all queries
def msvv(queries, new_budget, keyword_adv, adv_budget):
	revenue = 0.0
	all_spent = True
	#checks max_bid_value of all bidders for that query_advertiser
	for query in queries:
		#checks if all the neighbours have any budget left
		for adv_bid in keyword_adv[query]:
			if new_budget[adv_bid[0]] >= adv_bid[1]:
				all_spent = False	 
		#if any neighbour has some budget left , it then enters the following loop
		if(all_spent == False):
			
			max_msvv_val = sys.float_info.min
			max_bidder = -1
			max_bid = -1
			#checks bids for all advertisers for every query and selects the max scaledbid acc to the psi func
			for adv_bid in keyword_adv[query]:
				query_advertiser = adv_bid[0]
				advertiser_budget = new_budget[query_advertiser]
				bid = adv_bid[1]

				#calculates fraction of remaining budget
				xu = (adv_budget[query_advertiser] - advertiser_budget)/adv_budget[query_advertiser]
				#calculates psi function value for xu
				psi_xu = 1-np.exp(xu-1)

				#calculates the scaled msvv value
				msvv_val = bid*psi_xu
				
				#checks if budget of bidder for that particular query is more than the submitted bid
				if advertiser_budget >= bid:
					#checks if the msvv value of current bid is max or not and updates max
					if msvv_val > max_msvv_val:
						max_msvv_val = msvv_val
						max_bidder = query_advertiser
						max_bid = bid
					# if 2 bidder's have same msvv value then select the one with the least id
					if msvv_val == max_msvv_val:
						if query_advertiser < max_bidder:
							max_bidder = query_advertiser
							max_bid = bid
			#add the bid_value to the revenue and subtract bid value from bidder's budget
			#do this only if there is any bidder available with sufficient budget
			#if any of the above condition fails, the max_bid, and max_bidder wont get updated 
			if max_bidder != -1 and max_bid != -1: 
				revenue+= max_bid
				new_budget[max_bidder] -= max_bid

	return revenue

#Function used to compute the balance algorithm
#returns the revenue(float) for all queries
def balance(queries, new_budget, keyword_adv):
	revenue = 0.0
	all_spent = True
	#checks max_bid_value of all bidders for that query_advertiser
	for query in queries:
		#checks if all the neighbours have any budget left
		for adv_bid in keyword_adv[query]:
			if new_budget[adv_bid[0]] >= adv_bid[1]:
				all_spent = False	 
		#if any neighbour has some budget left , it then enters the following loop
		if(all_spent == False):
			
			max_budget = sys.float_info.min
			max_bidder = -1
			max_bid = -1
			#checks bids for all advertisers for every query and selects the max unspent budget advertiser bid
			for adv_bid in keyword_adv[query]:
				query_advertiser = adv_bid[0]
				advertiser_budget = new_budget[query_advertiser]
				bid = adv_bid[1]
				
				
				#checks if budget of bidder for that particular query is more than presented bid
				if advertiser_budget >= bid:
					#checks if the remaining budget from that bidder is max
					if advertiser_budget > max_budget:
						max_budget = advertiser_budget
						max_bidder = query_advertiser
						max_bid = bid
					# if 2 bidder's have same budget then select the one with the least id
					if advertiser_budget == max_budget:
						if query_advertiser < max_bidder:
							max_bidder = query_advertiser
							max_bid = bid
			#add the bid_value to the revenue and subtract bid value from bidder's budget
			#do this only if there is any bidder available with sufficient budget
			#if any of the above condition fails, the max_bid, and max_bidder wont get updated 
			if max_bidder != -1 and max_bid != -1: 
				revenue+= max_bid
				new_budget[max_bidder] -= max_bid

	return revenue 

def revenue_ratio(queries, adv_budget, keyword_adv, optimum_budget, algo):

	rev = 0.0

	#calculates the revenue value of queries in the given order
	new_budget = dict(adv_budget)
	#selects the function according to the input
	if algo =='greedy':
		rev = greedy(queries, new_budget, keyword_adv)
	elif algo =='msvv':
		rev = msvv(queries, new_budget, keyword_adv, dict(adv_budget))
	elif algo =='balance':
		rev = balance(queries, new_budget, keyword_adv)
	else:
		print('invalid input')
		sys.exit(0)
	
	print('revenue = ' + str(round(rev, 2)))

	total_revenue=0.0
	rev = 0.0
	permutations = 100
	random.seed(0)

	#calculates the competitive ratio
	for i in range(permutations):
		new_budget = dict(adv_budget)
		random.shuffle(queries)

		if algo =='greedy':
			rev = greedy(queries, new_budget, keyword_adv)
		elif algo =='msvv':
			rev = msvv(queries, new_budget, keyword_adv, dict(adv_budget))
		elif algo =='balance':
			rev = balance(queries, new_budget, keyword_adv)
		else:
			print('invalid input')
			break;
		total_revenue += rev

	mean_revenue = total_revenue/permutations
	ratio = mean_revenue/optimum_budget

	print('competitive ratio = ' + str(round(ratio, 2)))

if __name__ == '__main__':
	main()
