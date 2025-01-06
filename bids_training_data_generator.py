# Rikiki training data generator
import random
from treys import Card, Deck
from copy import deepcopy
import math
from functools import reduce

import pandas as pd
import numpy as np

# Issue
# The hand distribution is skewed towards smaller hands. Non-unique hands of 2 cards are much more (usually around 10x) common than unique hands of 10. A solution for this would be to get the chances of 2 unique hands for each hand size, and then normalize the values to the desired training data length. 

# Generate how many tries it takes on average to get the same hand for each hand sze
# chances = []

# for n in range(2,14):
#     chances.append(int(
#         math.factorial(52) / (math.factorial(n) * math.factorial(52-n)))
#     )

# gcd = reduce(math.gcd, chances)
# print(gcd)
# reduced = [item // gcd for item in chances]
# print (reduced)

# min_val = min(reduced)
# max_val = max(reduced)
# a = 1
# b = 10000 # iterations
# normalized = [int(a + (x - min_val) * (b - a) / (max_val - min_val)) for x in reduced]
# print(normalized)

# This however yields us with 1 hand for the first 5-6 hand size, and 10k hands for 13 hand size. 
# It would be nice to find some middle ground for this
# A real solution would be to have an input with sum(reduced) hands tried, but that would be in the 100 billion range

# Another approach is to train a model for each hand count.

# Generate the dictionary
ranks = "23456789TJQKA"
suits = "shdc"  # Spades, Hearts, Diamonds, Clubs
card_dict = {f"{rank}{suit}": i for i, (rank, suit) in enumerate((r, s) for s in suits for r in ranks)}

def from_str_to_index_in_bitmap(str):
    return card_dict[str]

def str_hand_to_bitmap (hand):
    bitmap = [0 for _ in range(52)]
    for card in hand:
        bitmap[from_str_to_index_in_bitmap(card)] = 1
    
    return bitmap

def int_hand_to_bitmap (hand):
    bitmap = [0 for _ in range(52)]
    for card in hand:
        bitmap[from_str_to_index_in_bitmap(Card.int_to_str(card))] = 1
    
    return bitmap


deck = Deck()
player_count=4

# cards is a bitmap. 1 -> card in hand, 0 -> card not in hand. Bid is the rounds won using this hand.
for hand_size in range(2,14):
    training_data = []
    for _ in range(100000): # do not go too high here
        hands = []
        bets = []
        for i in range(player_count):
            #everyone draws cards
            hands.append(deck.draw(hand_size))
        hands_copy = deepcopy(hands)

        starting_player = hand_size % player_count # Starting player always rotating at the end of round. Overwritten every turn.
        wins = [0 for _ in range(player_count)]
        for _ in range(hand_size):
            # All players play 1 card per turn
            # The last winner of the turn starts the next turn
            order_this_round = hands[starting_player:] + hands[:starting_player]
            played_cards = [] #this has the string format of the cards
            for hand in order_this_round:
                played_cards.append(Card.int_to_str(hand.pop(random.randrange(0, len(hand)))))
            
            adu = played_cards.pop(0)
            winner = 0 # adu player is default winner
            
            i=1
            for card in played_cards:
                
                if Card.CHAR_RANK_TO_INT_RANK[card[0]] > Card.CHAR_RANK_TO_INT_RANK[adu[0]] and Card.CHAR_SUIT_TO_INT_SUIT[card[1]] == Card.CHAR_SUIT_TO_INT_SUIT[adu[1]]:
                    adu = card
                    winner = i
                i+=1
            
            # Give the winner the turn
            wins[(winner + starting_player) % player_count] += 1
            starting_player = (winner + starting_player) % player_count
        # Add wins
        i = 0
        for hand,win in zip(hands_copy, wins):

            training_data.append([int_hand_to_bitmap(hand), win])

        deck = Deck()

    combined = np.array([[*row[0], row[1]] for row in training_data])

    df = pd.DataFrame(combined)
    names=[f"{rank}{suit}" for suit in suits for rank in ranks]
    names.append("bid")
    
    # Normalize the bids to 0-1.
    # This way, the "bid" will represent how many % of your hand will score.
    col_min = df.iloc[:, -1].min()
    col_max = df.iloc[:, -1].max()
    df.iloc[:, -1] = ((df.iloc[:, -1] - col_min) / (col_max - col_min)).astype('float64')

    df.to_csv(f"bid_data_{hand_size}.csv", header=names, index=False)
    # df.to_csv(f"bid_data_example.csv", header=names, index=False)


# features = np.array([row[0] for row in training_data])
# labels = np.array([row[1] for row in training_data])


