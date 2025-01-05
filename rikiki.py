# Rikiki game model
import random
from treys import Card, Deck

deck = Deck()
# hand_size = random.randint(2,10)
player_count=4
avg_scores = []
for _ in range(10000):
    scores = [0 for _ in range(player_count)]


    for hand_size in range(2,14): # Skip round one because it's played by a different ruleset, and is just for fun
        hands = []
        bets = []
        for i in range(player_count):
            #everyone draws cards and submits their bets
            hands.append(deck.draw(hand_size))
            # This is the first thing the AI should determine: how much to bet based on the cards in hand, the current hand size, and the player count. Now a normal distribution is used with a relatively low sigma to simulate a real game.
            bets.append(max(0, round(random.gauss(hand_size / player_count, 0.5)))) 
            
        # print("Round", hand_size, "Bets: ", bets, "Sum: ", sum(bets))
        # print("Deviation of bets: ", sum(bets) - hand_size)

        starting_player = hand_size % player_count # Starting player always rotating at the end of round. Overwritten every turn.
        wins = [0 for _ in range(player_count)]
        for _ in range(hand_size):
            # All players play 1 card per turn
            # The last winner of the turn starts the next turn
            order_this_round = hands[starting_player:] + hands[:starting_player]
            played_cards = [] #this has the string format of the cards
            for hand in order_this_round:
                # Second thing the AI should determine: which card to play based on
                # * played cards so far (NOTE: could be a 52-large bitmap?)
                # * number of players
                # * hand size
                # * cards remaining in hand
                # * number of bets submitted by each player -> their location around the table might also matter!
                # * number of own bets
                # * starting player this turn
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
        
        # Score players at end of round
        i = 0
        for win, bet in zip(wins, bets):
            if win == bet:
                scores[i]+= (10 + bet * 2)
            else:
                scores[i]-= abs(win - bet * 2)
            i+=1

        deck.shuffle()

    avg_scores.append(sum(scores) / len(scores))

print(sum(avg_scores) / len(avg_scores))
