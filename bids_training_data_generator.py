# Rikiki training data generator
import random
from treys import Card, Deck
from copy import deepcopy

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
# hand_size = random.randint(2,10)
player_count=4

# training_data = [["cards", "bid"]]
training_data = []
# cards is a bitmap. 1 -> card in hand, 0 -> card not in hand. Bid is the rounds won using this hand.
for _ in range(100000):
    for hand_size in range(2,14): # Skip round one because it's played by a different ruleset, and is just for fun
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

import pandas as pd
import numpy as np

features = np.array([row[0] for row in training_data])
labels = np.array([row[1] for row in training_data])

combined = np.array([[*row[0], row[1]] for row in training_data])

df = pd.DataFrame(combined)
df.to_csv("bid_data.csv", index=False, header=False)
