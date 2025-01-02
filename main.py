from treys import Card, Evaluator, Deck
import random
import collections
from math import ceil

deck = Deck()
evaluator = Evaluator()

hand = deck.draw(7)

hand_eval = evaluator.evaluate(hand, [])

print("The current hand is: ", evaluator.class_to_string(evaluator.get_rank_class(hand_eval)), "with score", hand_eval)
Card.print_pretty_cards(hand)

# if larger hand is needed, I can run the evaluator on more hands using combinatorics and pick the best resulting score
# OR better yet, fork the library and add as needed (doesnt seem hard)

scores = {}

for n in range(100000):
    #simulate n possible discard+draw scenarios
    discarded_amount = random.randint(1,5)
    hand_to_discard = random.sample(hand, discarded_amount)
    
    #print("Cards to discard are")
    #Card.print_pretty_cards(hand_to_discard)
    
    new_hand = [item for item in hand if item not in hand_to_discard]
    drawn_cards = deck.draw(discarded_amount)
    
    #print("Drawn cards are")
    #Card.print_pretty_cards(drawn_cards)
    
    new_hand = new_hand + drawn_cards
    #print("New hand is")
    #Card.print_pretty_cards(new_hand)
    
    # evaluate the new hand, and add its value to the map. Each discarded card batch is one key in the map, and the associated resulting scores from the simulation are in a list as a value.
    new_hand_eval = evaluator.evaluate([], new_hand)

    hand_to_discard = sorted(hand_to_discard)
    if tuple(hand_to_discard) not in scores:
        scores[tuple(hand_to_discard)] = []
    scores[tuple(hand_to_discard)].append(new_hand_eval)
    
    # Reshuffle drawn cards
    deck.cards = deck.cards + drawn_cards
    random.shuffle(deck.cards)

# Get the average scores of discards
averages={}
for key, value_list in scores.items():
    avg = sum(value_list) / len(value_list) if value_list else 0
    averages[key] = avg

# Get the best resulting score
min_key = min(averages, key=averages.get)
min_value = averages[min_key]

print("Discard")
Card.print_pretty_cards(list(min_key))
print("To get the most likely hand:", evaluator.class_to_string(evaluator.get_rank_class(hand_eval)), "with expected score", ceil(min_value))

# todo: simulate having multiple plays/discards and maximize score that way
