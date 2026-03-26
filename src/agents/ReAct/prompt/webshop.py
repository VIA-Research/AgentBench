def get_system_prompt(fewshots: int) -> str:
    webshop_instruction_react = """
Solve a shopping task with interleaving Thought and Action steps.

You will be given a shopping instruction and the current webshop observation.
You must respond in ReAct format only:
- exactly one `Thought:` line
- exactly one `Action:` line

Rules:
- Use one action at a time.
- Do not output numbered plans, bullet lists, multiple actions, or `Observation:`.
- Keep `Thought:` short and useful, about 1-2 sentences.
- Never say you do not know; use tools to gather the missing information.

Available actions:
- `search[query]`: use this only when the observation explicitly shows `[Search]`.
- `click[target]`: use this only for visible clickable targets that appear inside brackets, such as item ids, options, `[Next >]`, `[Back to Search]`, `[< Prev]`, `[Description]`, `[Features]`, `[Reviews]`, or `[Buy Now]`.

Navigation rules:
- After clicking `Description`, `Features`, `Reviews`, or `Attributes`, click `[< Prev]` to return to the item page.
- If you want to inspect a different product after opening an item page, click `[< Prev]` first to go back to the search results.
- If the observation shows an invalid action or an error, recover using the current page. Usually `click[Back to Search]` or `click[< Prev]` is the right move.

End condition:
- The task ends only after `Action: click[Buy Now]`.
- Do not invent `finish[...]` or any other tool.
- The score appears only after clicking `Buy Now`.

Here are some example trajectories:
"""
    examples = ["""
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Thought: search for diary free and apple variety pack of chips under $30
Action: search[dairy free and apple variety pack of chips under $30]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Thought: The second item matches the criteria as it is dairy-free, a variety pack, and priced below $30.
Action: click[B01KMHY5PG]
Observation:
[Back to Search]
[< Prev]
flavor name [Apple][Strawberry]
size [Pack of 6][Pack of 12]
Price: $21.49
Rating: 4.5 stars
[Description]
[Features]
[Reviews]
[Buy Now]

Thought: The product has the option 'Apple' which matches my preference. I will select it.
Action: click[Apple]
Observation: You have clicked Apple.


Thought: Now, I will click “Buy Now” to complete the purchase.
Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 0.5
Purchase successful with score 0.5
###
""",
"""
Instruction:
i need gluten free vegetarian smoked peppered bacon - 4 ounce (pack of 2), and price lower than 40.00 dollars
[Search]

Thought: Search for gluten-free vegetarian smoked peppered bacon 4-ounce pack of 2 under $40.
Action: search[gluten free vegetarian smoked peppered bacon 4 ounce pack of 2 under $40]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Thought: The first item partially matches the query but needs confirmation of the pack size.
Action: click[B06Y96MXJV]
Observation:
[Back to Search]
[< Prev]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
Options:
size [4 oz (Pack of 2)][12 oz (Pack of 1)]
Price: $29.99
Rating: 4.8 stars
[Description]
[Features]
[Reviews]
[Buy Now]

Thought: The product has the size '4 oz (Pack of 2)' and meets the gluten-free and price constraints.
Action: click[4 oz (Pack of 2)]
Observation: You have clicked 4 oz (Pack of 2).

Thought: Now, I will click “Buy Now” to complete the purchase.
Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 1.0
Purchase successful with score 1.0
###

Instruction:
i need vegan snack pack with a variety of flavors, and price lower than 15.00 dollars
[Search]

Thought: Search for a vegan snack pack with a variety of flavors under $15.
Action: search[vegan snack pack flavor variety under $15]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B09XYZ123]
Vegan Snack Pack Variety - $14.99
[B08ABCDE12]
Vegan Chips, Single Flavor - $10.00

Thought: The first item matches the criteria as it is vegan, includes a flavor variety, and is priced at $14.99.
Action: click[B09XYZ123]
Observation:
[Back to Search]
[< Prev]
Options:
flavor name [Flavor A][Flavor B]
size [Variety Pack]
Price: $14.99
Rating: 4.6 stars
[Description]
[Features]
[Reviews]
[Buy Now]

Thought: The product has the 'Variety Pack' option and meets the price constraint.
Action: click[Variety Pack]
Observation: You have clicked Variety Pack.

Thought: Now, I will click “Buy Now” to complete the purchase.
Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 0.9
Purchase successful with score 0.9
###
"""]
    return webshop_instruction_react + "\n".join(examples[:fewshots]) + "Now you need to answer\nThought:\nAction:"
