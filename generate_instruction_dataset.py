import pandas as pd
import json
import ast
from datetime import datetime
from collections import defaultdict, Counter
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class InstructionDatasetGenerator:
    def __init__(self, recipes_path, reviews_path):
        """Initialize the generator with dataset paths."""
        print("Loading datasets...")
        self.recipes_df = pd.read_csv(recipes_path, low_memory=False)
        self.reviews_df = pd.read_csv(reviews_path, low_memory=False)

        # Clean column names
        self.recipes_df.columns = self.recipes_df.columns.str.strip()
        self.reviews_df.columns = self.reviews_df.columns.str.strip()

        print(f"Loaded {len(self.recipes_df)} recipes and {len(self.reviews_df)} reviews")

        # Parse dates (CRITICAL for temporal tasks)
        self.reviews_df['submit_date'] = pd.to_datetime(self.reviews_df['submit_date'], format='%d/%m/%Y', errors='coerce')

        # Group reviews by user
        self.user_reviews = self.reviews_df.groupby('member_id')

        # Create recipe lookup
        self.recipe_lookup = self.recipes_df.set_index('recipe_id').to_dict('index')

        # Storage for generated instructions
        self.instructions = []

        # Track unique instruction inputs to avoid redundancy
        self.seen_instruction_hashes = set()

    def is_unique_instruction(self, instruction_input):
        """Check if instruction is unique to avoid redundancy."""
        # Create hash of the instruction input
        instruction_hash = hash(instruction_input)
        if instruction_hash in self.seen_instruction_hashes:
            return False
        self.seen_instruction_hashes.add(instruction_hash)
        return True

    def parse_tags(self, tags_str):
        """Parse tags from string format."""
        if pd.isna(tags_str) or tags_str == '':
            return []
        try:
            tags = ast.literal_eval(tags_str)
            return tags if isinstance(tags, list) else []
        except:
            return []

    def parse_ingredients(self, ingredients_str):
        """Parse ingredients from string format."""
        if pd.isna(ingredients_str):
            return []
        try:
            ingredients_dict = ast.literal_eval(ingredients_str)
            if isinstance(ingredients_dict, dict):
                ingredients = []
                for key, items in ingredients_dict.items():
                    for item in items:
                        if isinstance(item, tuple) and len(item) >= 1:
                            ingredients.append(item[0])
                return ingredients
        except:
            return []
        return []

    def get_recipe_info(self, recipe_id):
        """Get detailed recipe information."""
        if recipe_id not in self.recipe_lookup:
            return None

        recipe = self.recipe_lookup[recipe_id]
        tags = self.parse_tags(recipe.get('tags', ''))
        ingredients = self.parse_ingredients(recipe.get('ingredients', ''))

        # Get health scores
        fsa_score = recipe.get('fsa_score', np.nan)
        nutri_score = recipe.get('nutri_score', np.nan)
        who_score = recipe.get('who_score', np.nan)

        # Determine health label using combination of scores
        # Priority: FSA > NutriScore > WHO
        health_scores = []
        if pd.notna(fsa_score):
            health_scores.append(fsa_score)
        if pd.notna(nutri_score):
            health_scores.append(nutri_score)
        if pd.notna(who_score):
            health_scores.append(who_score)

        if health_scores:
            avg_health_score = np.mean(health_scores)
            if avg_health_score <= 0.33:
                health_label = "healthy"
                fsa_color = "green"
            elif avg_health_score <= 0.66:
                health_label = "moderate"
                fsa_color = "amber"
            else:
                health_label = "unhealthy"
                fsa_color = "red"
        else:
            health_label = "unknown"
            fsa_color = "unknown"

        # Add detailed health description
        if pd.notna(fsa_score):
            health_label_detailed = f"{health_label} (FSA: {fsa_color})"
        else:
            health_label_detailed = health_label

        return {
            'recipe_id': recipe_id,
            'title': recipe.get('title', 'Unknown'),
            'tags': tags,
            'ingredients': ingredients,
            'health_label': health_label_detailed,
            'health_label_simple': health_label,
            'fsa_score': fsa_score,
            'nutri_score': nutri_score,
            'who_score': who_score,
            'calories': recipe.get('calories [cal]', 0),
            'protein': recipe.get('protein [g]', 0),
            'totalFat': recipe.get('totalFat [g]', 0),
            'sodium': recipe.get('sodium [mg]', 0),
            'average_rating': recipe.get('average_rating', 0)
        }

    def map_rating_to_preference(self, rating):
        """Map numeric rating to preference class."""
        if rating >= 5:
            return "Strongly Recommend"
        elif rating >= 4:
            return "Recommend"
        elif rating >= 3:
            return "Neutral"
        else:
            return "Do Not Recommend"

    def format_recipe_context(self, recipe_info, include_health=True):
        """Format recipe information for instruction input."""
        if not recipe_info:
            return "Unknown recipe"

        tags_str = ", ".join(recipe_info['tags'][:5]) if recipe_info['tags'] else "no tags"
        ingredients_str = ", ".join(recipe_info['ingredients'][:8]) if recipe_info['ingredients'] else "no ingredients"

        context = f"- '{recipe_info['title']}'\n"
        context += f"  Tags: {tags_str}\n"
        context += f"  Ingredients: {ingredients_str}\n"

        if include_health:
            context += f"  Health: {recipe_info['health_label']}"

        return context

    # ========== TASK 1: Personalized Preference Simulation ==========
    def generate_task1_preference_simulation(self, max_samples=10000):
        """Task 1: Learn my taste - Contextualized decision tasks."""
        print("\n=== Generating Task 1: Personalized Preference Simulation ===")
        task1_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 3:
                continue

            # Get liked and disliked recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 1 or len(disliked) < 1:
                continue

            # Sample target recipe
            target_review = user_data.sample(1).iloc[0]
            target_recipe = self.get_recipe_info(target_review['recipe_id'])

            if not target_recipe:
                continue

            # Sample context recipes
            liked_sample = liked.sample(min(2, len(liked)))
            disliked_sample = disliked.sample(min(2, len(disliked)))

            # Build instruction
            instruction_input = "User liked:\n"
            for _, review in liked_sample.iterrows():
                recipe_info = self.get_recipe_info(review['recipe_id'])
                if recipe_info:
                    instruction_input += self.format_recipe_context(recipe_info) + "\n\n"

            instruction_input += "User disliked:\n"
            for _, review in disliked_sample.iterrows():
                recipe_info = self.get_recipe_info(review['recipe_id'])
                if recipe_info:
                    instruction_input += self.format_recipe_context(recipe_info) + "\n\n"

            instruction_input += "Target recipe:\n"
            instruction_input += self.format_recipe_context(target_recipe) + "\n\n"
            instruction_input += "Question: Would the user recommend the target recipe?"

            instruction_output = self.map_rating_to_preference(target_review['rating'])

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task1_instructions.append({
                    "task_type": "Task_1_Preference_Simulation",
                    "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task1_instructions) >= max_samples:
                break

        print(f"Generated {len(task1_instructions)} Task 1 instructions")
        return task1_instructions

    # ========== TASK 2: Pairwise Preference with Context ==========
    def generate_task2_pairwise_preference(self, max_samples=10000):
        """Task 2: Which of these two recipes will the user like more?"""
        print("\n=== Generating Task 2: Pairwise Preference ===")
        task2_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 4:
                continue

            # Get liked and disliked recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 2 or len(disliked) < 1:
                continue

            # Sample two recipes with different ratings
            sorted_reviews = user_data.sort_values('rating', ascending=False)
            if len(sorted_reviews) < 2:
                continue

            recipe_a_review = sorted_reviews.iloc[0]
            recipe_b_review = sorted_reviews.iloc[-1]

            if recipe_a_review['rating'] == recipe_b_review['rating']:
                continue

            recipe_a = self.get_recipe_info(recipe_a_review['recipe_id'])
            recipe_b = self.get_recipe_info(recipe_b_review['recipe_id'])

            if not recipe_a or not recipe_b:
                continue

            # Build context
            context_liked = liked[~liked['recipe_id'].isin([recipe_a_review['recipe_id'], recipe_b_review['recipe_id']])].sample(min(2, len(liked)-1))
            context_disliked = disliked.sample(min(1, len(disliked)))

            instruction_input = "User liked:\n"
            for _, review in context_liked.iterrows():
                recipe_info = self.get_recipe_info(review['recipe_id'])
                if recipe_info:
                    instruction_input += self.format_recipe_context(recipe_info) + "\n\n"

            instruction_input += "User disliked:\n"
            for _, review in context_disliked.iterrows():
                recipe_info = self.get_recipe_info(review['recipe_id'])
                if recipe_info:
                    instruction_input += self.format_recipe_context(recipe_info) + "\n\n"

            instruction_input += "Compare:\n"
            instruction_input += f"- Recipe A: {self.format_recipe_context(recipe_a)}\n"
            instruction_input += f"- Recipe B: {self.format_recipe_context(recipe_b)}\n\n"
            instruction_input += "Question: Which recipe is the user more likely to prefer?"

            # Output the higher-rated recipe
            if recipe_a_review['rating'] > recipe_b_review['rating']:
                instruction_output = f"Recipe A: {recipe_a['title']}"
            else:
                instruction_output = f"Recipe B: {recipe_b['title']}"

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task2_instructions.append({
                    "task_type": "Task_2_Pairwise_Preference",
                    "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task2_instructions) >= max_samples:
                break

        print(f"Generated {len(task2_instructions)} Task 2 instructions")
        return task2_instructions

    # ========== TASK 3: Tag-Based Future Preference ==========
    def generate_task3_tag_based_preference(self, max_samples=10000):
        """Task 3: Predict preference based on tag-level affinities."""
        print("\n=== Generating Task 3: Tag-Based Preference ===")
        task3_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 3:
                continue

            # Collect tags from liked and disliked recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 1 or len(disliked) < 1:
                continue

            liked_tags = []
            for _, review in liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe:
                    liked_tags.extend(recipe['tags'])

            disliked_tags = []
            for _, review in disliked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe:
                    disliked_tags.extend(recipe['tags'])

            if not liked_tags or not disliked_tags:
                continue

            # Get most common tags
            liked_tag_counts = Counter(liked_tags)
            disliked_tag_counts = Counter(disliked_tags)

            top_liked_tags = [tag for tag, _ in liked_tag_counts.most_common(5)]
            top_disliked_tags = [tag for tag, _ in disliked_tag_counts.most_common(5)]

            # Pick a candidate recipe
            candidate_review = user_data.sample(1).iloc[0]
            candidate_recipe = self.get_recipe_info(candidate_review['recipe_id'])

            if not candidate_recipe:
                continue

            instruction_input = "User's tag preferences:\n"
            instruction_input += f"- Liked tags: {', '.join(top_liked_tags)}\n"
            instruction_input += f"- Disliked tags: {', '.join(top_disliked_tags)}\n\n"
            instruction_input += "Candidate recipe:\n"
            instruction_input += self.format_recipe_context(candidate_recipe, include_health=False) + "\n\n"
            instruction_input += "Question: Will the user like this recipe?"

            instruction_output = self.map_rating_to_preference(candidate_review['rating'])

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task3_instructions.append({
                "task_type": "Task_3_Tag_Based_Preference",
                "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task3_instructions) >= max_samples:
                break

        print(f"Generated {len(task3_instructions)} Task 3 instructions")
        return task3_instructions

    # ========== TASK 4: Sequential Session Simulation ==========
    def generate_task4_sequential_session(self, max_samples=10000):
        """Task 4: Predict the next recipe in a browsing/rating sequence."""
        print("\n=== Generating Task 4: Sequential Session ===")
        task4_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 5:
                continue

            # Sort by submit_date timestamp (CRITICAL for temporal dynamics)
            sorted_reviews = user_data.sort_values('submit_date', na_position='last')
            sorted_reviews = sorted_reviews.dropna(subset=['submit_date'])

            if len(sorted_reviews) < 5:
                continue

            # Generate multiple sequences from this user's history
            # Take last 2-3 reviews as context and next as target
            for i in range(2, min(len(sorted_reviews) - 1, 10)):  # Limit to 10 sequences per user
                context_size = random.choice([2, 3])
                context_reviews = sorted_reviews.iloc[max(0, i-context_size):i]
                target_review = sorted_reviews.iloc[i]

                # Validate that we have valid recipes
                context_recipes = []
                for _, review in context_reviews.iterrows():
                    recipe = self.get_recipe_info(review['recipe_id'])
                    if recipe:
                        context_recipes.append((recipe, review))

                if len(context_recipes) < 2:
                    continue

                target_recipe = self.get_recipe_info(target_review['recipe_id'])
                if not target_recipe:
                    continue

                # Build instruction with temporal context
                instruction_input = "User previously rated (in chronological order):\n"
                for idx, (recipe, review) in enumerate(context_recipes, 1):
                    tags_str = ", ".join(recipe['tags'][:3]) if recipe['tags'] else "no tags"
                    instruction_input += f"{idx}. '{recipe['title']}' (rating {int(review['rating'])}, tags: {tags_str})\n"

                instruction_input += "\nQuestion: Based on this browsing pattern, what recipe did the user try next?"

                instruction_output = target_recipe['title']

                # Check for uniqueness to avoid redundancy
                if self.is_unique_instruction(instruction_input):
                    task4_instructions.append({
                    "task_type": "Task_4_Sequential_Session",
                    "instruction_input": instruction_input,
                        "instruction_output": instruction_output
                    })

                if len(task4_instructions) >= max_samples:
                    break

            if len(task4_instructions) >= max_samples:
                break

        print(f"Generated {len(task4_instructions)} Task 4 instructions")
        return task4_instructions

    # ========== TASK 5: Session Preference Drift ==========
    def generate_task5_preference_drift(self, max_samples=10000):
        """Task 5: Did the user's taste shift during this session?"""
        print("\n=== Generating Task 5: Session Preference Drift ===")
        task5_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 5:
                continue

            # Sort by submit_date timestamp (CRITICAL for temporal dynamics)
            sorted_reviews = user_data.sort_values('submit_date', na_position='last')
            sorted_reviews = sorted_reviews.dropna(subset=['submit_date'])

            if len(sorted_reviews) < 5:
                continue

            # Use sliding window approach to detect drift
            # Generate multiple drift examples per user
            for start_idx in range(0, min(len(sorted_reviews) - 5, 8)):  # Multiple windows per user
                # Split into early and later session (with temporal gap)
                early_session = sorted_reviews.iloc[start_idx:start_idx+3]
                later_session = sorted_reviews.iloc[start_idx+3:start_idx+6]

                if len(later_session) == 0:
                    continue

                # Pick a candidate from later session
                candidate_review = later_session.iloc[random.randint(0, len(later_session)-1)]
                candidate_recipe = self.get_recipe_info(candidate_review['recipe_id'])

                if not candidate_recipe:
                    continue

                # Build instruction showing temporal evolution
                instruction_input = "Early in the session, the user rated (in chronological order):\n"
                early_recipes = []
                for idx, (_, review) in enumerate(early_session.iterrows(), 1):
                    recipe = self.get_recipe_info(review['recipe_id'])
                    if recipe:
                        tags_str = ", ".join(recipe['tags'][:3]) if recipe['tags'] else "no tags"
                        instruction_input += f"{idx}. '{recipe['title']}' (rating {int(review['rating'])}, tags: {tags_str}, {recipe['health_label']})\n"
                        early_recipes.append(recipe)

                if len(early_recipes) < 2:
                    continue

                instruction_input += "\nLater in the session, they encountered:\n"
                tags_str = ", ".join(candidate_recipe['tags'][:3]) if candidate_recipe['tags'] else "no tags"
                instruction_input += f"- '{candidate_recipe['title']}' (tags: {tags_str}, {candidate_recipe['health_label']})\n\n"
                instruction_input += "Question: How did the user rate this later recipe?"

                instruction_output = self.map_rating_to_preference(candidate_review['rating'])

                # Check for uniqueness to avoid redundancy
                if self.is_unique_instruction(instruction_input):
                    task5_instructions.append({
                    "task_type": "Task_5_Session_Drift",
                    "instruction_input": instruction_input,
                        "instruction_output": instruction_output
                    })

                if len(task5_instructions) >= max_samples:
                    break

            if len(task5_instructions) >= max_samples:
                break

        print(f"Generated {len(task5_instructions)} Task 5 instructions")
        return task5_instructions

    # ========== TASK 6: Session Repetition vs Exploration ==========
    def generate_task6_repetition_exploration(self, max_samples=10000):
        """Task 6: Does the user stick with familiar categories or explore?"""
        print("\n=== Generating Task 6: Repetition vs Exploration ===")
        task6_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 6:
                continue

            # Sort by submit_date timestamp (CRITICAL for temporal dynamics)
            sorted_reviews = user_data.sort_values('submit_date', na_position='last')
            sorted_reviews = sorted_reviews.dropna(subset=['submit_date'])

            if len(sorted_reviews) < 4:
                continue

            # Find 2-3 consecutive recipes and check the next one (temporal sequence matters!)
            for i in range(min(len(sorted_reviews) - 3, 10)):  # Multiple sequences per user
                session_recipes = []
                for j in range(i, min(i+3, len(sorted_reviews))):
                    review = sorted_reviews.iloc[j]
                    recipe = self.get_recipe_info(review['recipe_id'])
                    if recipe:
                        session_recipes.append((review, recipe))

                if len(session_recipes) < 2:
                    continue

                # Get tags from first 2 recipes (temporal context)
                session_tags = []
                for _, recipe in session_recipes[:2]:
                    session_tags.extend(recipe['tags'])

                # Get the next recipe (temporal prediction)
                if i + 2 < len(sorted_reviews):
                    next_review = sorted_reviews.iloc[i + 2]
                    next_recipe = self.get_recipe_info(next_review['recipe_id'])

                    if not next_recipe:
                        continue

                    # Check if exploration or repetition based on tag overlap
                    next_tags = set(next_recipe['tags'])
                    session_tags_set = set(session_tags)

                    # If shares many tags, it's repetition; otherwise exploration
                    overlap = len(next_tags & session_tags_set)

                    # More nuanced labeling
                    if overlap >= 3:
                        label = "Repetition"
                    elif overlap >= 1:
                        label = "Partial Repetition"
                    else:
                        label = "Exploration"

                    # Simplify to binary for now
                    if overlap >= 2:
                        label = "Repetition"
                    else:
                        label = "Exploration"

                    instruction_input = "In this session, the user rated (in chronological order):\n"
                    for idx, (review, recipe) in enumerate(session_recipes[:2], 1):
                        tags_str = ", ".join(recipe['tags'][:3]) if recipe['tags'] else "no tags"
                        instruction_input += f"{idx}. '{recipe['title']}' ({tags_str}) – rating {int(review['rating'])}\n"

                    next_tags_str = ", ".join(next_recipe['tags'][:3]) if next_recipe['tags'] else "no tags"
                    instruction_input += f"\nNext recipe in session:\n- '{next_recipe['title']}' ({next_tags_str})\n\n"
                    instruction_input += "Question: Did the user continue with the same category or explore a new one?"

                    # Check for uniqueness to avoid redundancy
                    if self.is_unique_instruction(instruction_input):
                        task6_instructions.append({
                        "task_type": "Task_6_Repetition_Exploration",
                        "instruction_input": instruction_input,
                            "instruction_output": label
                        })

                    if len(task6_instructions) >= max_samples:
                        break

            if len(task6_instructions) >= max_samples:
                break

        print(f"Generated {len(task6_instructions)} Task 6 instructions")
        return task6_instructions

    # ========== TASK 7: Preference-Aware Diversity ==========
    def generate_task7_diversity_queries(self, max_samples=10000):
        """Task 7: Not just pasta! – Generate diverse alternatives."""
        print("\n=== Generating Task 7: Diversity Queries ===")
        task7_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 5:
                continue

            # Sort by timestamp (temporal context)
            sorted_reviews = user_data.sort_values('submit_date', na_position='last')
            sorted_reviews = sorted_reviews.dropna(subset=['submit_date'])

            # Collect all tags from liked recipes
            liked = sorted_reviews[sorted_reviews['rating'] >= 4]

            if len(liked) < 3:
                continue

            # Find dominant tags (user's typical preferences)
            all_tags = []
            for _, review in liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and recipe['tags']:
                    all_tags.extend(recipe['tags'])

            if not all_tags:
                continue

            tag_counts = Counter(all_tags)
            dominant_tags = [tag for tag, _ in tag_counts.most_common(5)]

            # Find recipes outside dominant category (diverse alternatives)
            diverse_recipes = []
            for _, review in sorted_reviews.iterrows():
                if review['rating'] >= 4:
                    recipe = self.get_recipe_info(review['recipe_id'])
                    if recipe and recipe['tags']:
                        recipe_tags_set = set(recipe['tags'])
                        dominant_tags_set = set(dominant_tags)
                        # Low overlap means diversity
                        if len(recipe_tags_set & dominant_tags_set) < 2:
                            diverse_recipes.append(recipe)

            if len(diverse_recipes) < 1:
                continue

            # Sample dominant recipes for context
            dominant_recipes = []
            for _, review in liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and recipe['tags']:
                    recipe_tags_set = set(recipe['tags'])
                    dominant_tags_set = set(dominant_tags)
                    # High overlap means typical preference
                    if len(recipe_tags_set & dominant_tags_set) >= 2:
                        dominant_recipes.append(recipe)

            if len(dominant_recipes) < 2:
                continue

            instruction_input = f"The user mostly enjoys recipes with tags: {', '.join(dominant_tags[:3])} (liked "
            instruction_input += ", ".join([f"'{r['title']}'" for r in dominant_recipes[:3]])
            instruction_input += ").\n\nQuestion: Suggest diverse cuisines they might also enjoy."

            diverse_titles = [r['title'] for r in diverse_recipes[:2]]
            instruction_output = ", ".join(diverse_titles)

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task7_instructions.append({
                "task_type": "Task_7_Diversity_Queries",
                "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task7_instructions) >= max_samples:
                break

        print(f"Generated {len(task7_instructions)} Task 7 instructions")
        return task7_instructions

    # ========== TASK 8: FlavorMemory - Long-Term Taste Profiles ==========
    def generate_task8_flavor_memory(self, max_samples=10000):
        """Task 8: Track persistent flavor biases across long-term history."""
        print("\n=== Generating Task 8: FlavorMemory ===")
        task8_instructions = []

        # Simple flavor taxonomy based on ingredients
        flavor_profiles = {
            'umami': ['soy sauce', 'mushroom', 'beef', 'tomato', 'cheese', 'miso', 'anchovy'],
            'sweet': ['sugar', 'honey', 'chocolate', 'maple syrup', 'fruit', 'vanilla', 'caramel'],
            'spicy': ['chili', 'pepper', 'jalapeno', 'cayenne', 'hot sauce', 'curry', 'ginger'],
            'salty': ['salt', 'soy sauce', 'bacon', 'ham', 'olives', 'anchovies', 'pickle'],
            'sour': ['lemon', 'lime', 'vinegar', 'yogurt', 'sour cream', 'citrus', 'tomato']
        }

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 5:
                continue

            # Analyze flavor profiles from liked recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 3:
                continue

            # Count flavor profiles in liked recipes
            liked_flavors = defaultdict(int)
            for _, review in liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and recipe['ingredients']:
                    ingredients_lower = [str(ing).lower() for ing in recipe['ingredients'] if isinstance(ing, str)]
                    for flavor, keywords in flavor_profiles.items():
                        if any(keyword in ' '.join(ingredients_lower) for keyword in keywords):
                            liked_flavors[flavor] += 1

            if not liked_flavors:
                continue

            # Get dominant flavor
            dominant_flavor = max(liked_flavors, key=liked_flavors.get)

            # Pick a candidate recipe
            candidate_review = user_data.sample(1).iloc[0]
            candidate_recipe = self.get_recipe_info(candidate_review['recipe_id'])

            if not candidate_recipe:
                continue

            # Determine candidate flavor profile
            candidate_ingredients_lower = [str(ing).lower() for ing in candidate_recipe['ingredients'] if isinstance(ing, str)]
            candidate_flavors = []
            for flavor, keywords in flavor_profiles.items():
                if any(keyword in ' '.join(candidate_ingredients_lower) for keyword in keywords):
                    candidate_flavors.append(flavor)

            candidate_flavor_str = ", ".join(candidate_flavors[:2]) if candidate_flavors else "neutral"

            instruction_input = "User's long-term profile:\n"
            instruction_input += f"- Mostly likes {dominant_flavor}-rich recipes\n"
            if len(disliked) > 0:
                instruction_input += "- Sometimes dislikes overly rich desserts.\n"
            instruction_input += f"\nCandidate recipe: '{candidate_recipe['title']}' ({candidate_flavor_str} profile).\n\n"
            instruction_input += "Question: How did the user rate this recipe?"

            instruction_output = self.map_rating_to_preference(candidate_review['rating'])

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task8_instructions.append({
                "task_type": "Task_8_FlavorMemory",
                "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task8_instructions) >= max_samples:
                break

        print(f"Generated {len(task8_instructions)} Task 8 instructions")
        return task8_instructions

    # ========== TASK 9: Health Profile-Based Prediction ==========
    def generate_task9_health_profile(self, max_samples=10000):
        """Task 9: Align preference with user's health orientation."""
        print("\n=== Generating Task 9: Health Profile ===")
        task9_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 4:
                continue

            # Get liked and disliked recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 2 or len(disliked) < 1:
                continue

            # Compute average health metrics
            liked_fsa_scores = []
            liked_nutri_scores = []
            for _, review in liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and pd.notna(recipe['fsa_score']):
                    liked_fsa_scores.append(recipe['fsa_score'])
                if recipe and pd.notna(recipe['nutri_score']):
                    liked_nutri_scores.append(recipe['nutri_score'])

            disliked_fsa_scores = []
            for _, review in disliked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and pd.notna(recipe['fsa_score']):
                    disliked_fsa_scores.append(recipe['fsa_score'])

            if not liked_fsa_scores:
                continue

            avg_liked_fsa = np.mean(liked_fsa_scores)

            # Determine health orientation
            if avg_liked_fsa <= 0.3:
                health_orientation = "mostly NutriScore A/B recipes (soups, salads)"
            elif avg_liked_fsa <= 0.6:
                health_orientation = "moderate health recipes"
            else:
                health_orientation = "indulgent recipes"

            # Pick a candidate recipe
            candidate_review = user_data.sample(1).iloc[0]
            candidate_recipe = self.get_recipe_info(candidate_review['recipe_id'])

            if not candidate_recipe:
                continue

            # Determine candidate NutriScore label
            if pd.notna(candidate_recipe['nutri_score']):
                if candidate_recipe['nutri_score'] <= 0.2:
                    nutri_label = "A"
                elif candidate_recipe['nutri_score'] <= 0.4:
                    nutri_label = "B"
                elif candidate_recipe['nutri_score'] <= 0.6:
                    nutri_label = "C"
                else:
                    nutri_label = "D/E"
            else:
                nutri_label = "Unknown"

            instruction_input = "User's health profile:\n"
            instruction_input += f"- Liked: {health_orientation}\n"
            if disliked_fsa_scores:
                instruction_input += "- Disliked: fried, high-fat recipes.\n"
            instruction_input += f"\nCandidate recipe: '{candidate_recipe['title']}'\n"
            instruction_input += f"Calories: {int(candidate_recipe['calories'])}, NutriScore: {nutri_label}, Health: {candidate_recipe['health_label']}\n\n"
            instruction_input += "Question: Based on this health profile, how did the user rate this recipe?"

            instruction_output = self.map_rating_to_preference(candidate_review['rating'])

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task9_instructions.append({
                "task_type": "Task_9_Health_Profile",
                "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task9_instructions) >= max_samples:
                break

        print(f"Generated {len(task9_instructions)} Task 9 instructions")
        return task9_instructions

    # ========== TASK 10: Health vs Indulgence Trade-off ==========
    def generate_task10_health_tradeoff(self, max_samples=10000):
        """Task 10: When forced to choose, does the user prioritize taste or health?"""
        print("\n=== Generating Task 10: Health vs Indulgence ===")
        task10_instructions = []

        for member_id, user_data in self.user_reviews:
            if len(user_data) < 4:
                continue

            # Get context recipes
            liked = user_data[user_data['rating'] >= 4]
            disliked = user_data[user_data['rating'] <= 2]

            if len(liked) < 2 or len(disliked) < 1:
                continue

            # Find one healthy and one unhealthy recipe the user rated
            healthy_candidates = []
            unhealthy_candidates = []

            for _, review in user_data.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe and pd.notna(recipe['fsa_score']):
                    if recipe['fsa_score'] <= 0.3:
                        healthy_candidates.append((review, recipe))
                    elif recipe['fsa_score'] >= 0.6:
                        unhealthy_candidates.append((review, recipe))

            if len(healthy_candidates) < 1 or len(unhealthy_candidates) < 1:
                continue

            # Sample one from each
            healthy_review, healthy_recipe = random.choice(healthy_candidates)
            unhealthy_review, unhealthy_recipe = random.choice(unhealthy_candidates)

            # Build context
            context_liked = liked.sample(min(2, len(liked)))
            context_disliked = disliked.sample(min(1, len(disliked)))

            instruction_input = "User's previously rated recipes:\n"
            for _, review in context_liked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe:
                    tags_str = ", ".join(recipe['tags'][:3]) if recipe['tags'] else "no tags"
                    instruction_input += f"- Liked: '{recipe['title']}' ({recipe['health_label']}, tags: {tags_str})\n"

            for _, review in context_disliked.iterrows():
                recipe = self.get_recipe_info(review['recipe_id'])
                if recipe:
                    tags_str = ", ".join(recipe['tags'][:3]) if recipe['tags'] else "no tags"
                    instruction_input += f"- Disliked: '{recipe['title']}' ({recipe['health_label']}, tags: {tags_str})\n"

            instruction_input += "\nCompare recipes:\n"
            healthy_nutri = "A/B" if healthy_recipe['nutri_score'] <= 0.4 else "C"
            unhealthy_nutri = "D/E" if unhealthy_recipe['nutri_score'] >= 0.6 else "C"

            healthy_ingredients_str = ", ".join(healthy_recipe['ingredients'][:5])
            unhealthy_ingredients_str = ", ".join(unhealthy_recipe['ingredients'][:5])

            instruction_input += f"- Recipe A: '{healthy_recipe['title']}' (NutriScore {healthy_nutri}, healthy, ingredients: {healthy_ingredients_str})\n"
            instruction_input += f"- Recipe B: '{unhealthy_recipe['title']}' (NutriScore {unhealthy_nutri}, indulgent, ingredients: {unhealthy_ingredients_str})\n\n"
            instruction_input += "Question: Which recipe did the user rate higher?"

            # Output the higher-rated recipe
            if healthy_review['rating'] > unhealthy_review['rating']:
                instruction_output = f"Recipe A: {healthy_recipe['title']}"
            elif unhealthy_review['rating'] > healthy_review['rating']:
                instruction_output = f"Recipe B: {unhealthy_recipe['title']}"
            else:
                instruction_output = "Both rated equally"

            # Check for uniqueness to avoid redundancy
            if self.is_unique_instruction(instruction_input):
                task10_instructions.append({
                    "task_type": "Task_10_Health_Tradeoff",
                    "instruction_input": instruction_input,
                    "instruction_output": instruction_output
                })

            if len(task10_instructions) >= max_samples:
                break

        print(f"Generated {len(task10_instructions)} Task 10 instructions")
        return task10_instructions

    # ========== Main Generation Method ==========
    def generate_all_tasks(self, samples_per_task=10000, save_individual=True, output_dir="."):
        """Generate all 10 tasks and optionally save each task separately."""
        print("=" * 80)
        print("INSTRUCTION DATASET GENERATION")
        print("=" * 80)

        all_instructions = []
        task_results = {}

        # Generate each task
        print("\n[1/10] Generating Task 1...")
        task1 = self.generate_task1_preference_simulation(samples_per_task)
        task_results['Task_1_Preference_Simulation'] = task1
        all_instructions.extend(task1)
        if save_individual:
            self.save_task_to_json(task1, f"{output_dir}/task_1_preference_simulation.json")

        print("\n[2/10] Generating Task 2...")
        task2 = self.generate_task2_pairwise_preference(samples_per_task)
        task_results['Task_2_Pairwise_Preference'] = task2
        all_instructions.extend(task2)
        if save_individual:
            self.save_task_to_json(task2, f"{output_dir}/task_2_pairwise_preference.json")

        print("\n[3/10] Generating Task 3...")
        task3 = self.generate_task3_tag_based_preference(samples_per_task)
        task_results['Task_3_Tag_Based_Preference'] = task3
        all_instructions.extend(task3)
        if save_individual:
            self.save_task_to_json(task3, f"{output_dir}/task_3_tag_based_preference.json")

        print("\n[4/10] Generating Task 4...")
        task4 = self.generate_task4_sequential_session(samples_per_task)
        task_results['Task_4_Sequential_Session'] = task4
        all_instructions.extend(task4)
        if save_individual:
            self.save_task_to_json(task4, f"{output_dir}/task_4_sequential_session.json")

        print("\n[5/10] Generating Task 5...")
        task5 = self.generate_task5_preference_drift(samples_per_task)
        task_results['Task_5_Session_Drift'] = task5
        all_instructions.extend(task5)
        if save_individual:
            self.save_task_to_json(task5, f"{output_dir}/task_5_session_drift.json")

        print("\n[6/10] Generating Task 6...")
        task6 = self.generate_task6_repetition_exploration(samples_per_task)
        task_results['Task_6_Repetition_Exploration'] = task6
        all_instructions.extend(task6)
        if save_individual:
            self.save_task_to_json(task6, f"{output_dir}/task_6_repetition_exploration.json")

        print("\n[7/10] Generating Task 7...")
        task7 = self.generate_task7_diversity_queries(samples_per_task)
        task_results['Task_7_Diversity_Queries'] = task7
        all_instructions.extend(task7)
        if save_individual:
            self.save_task_to_json(task7, f"{output_dir}/task_7_diversity_queries.json")

        print("\n[8/10] Generating Task 8...")
        task8 = self.generate_task8_flavor_memory(samples_per_task)
        task_results['Task_8_FlavorMemory'] = task8
        all_instructions.extend(task8)
        if save_individual:
            self.save_task_to_json(task8, f"{output_dir}/task_8_flavor_memory.json")

        print("\n[9/10] Generating Task 9...")
        task9 = self.generate_task9_health_profile(samples_per_task)
        task_results['Task_9_Health_Profile'] = task9
        all_instructions.extend(task9)
        if save_individual:
            self.save_task_to_json(task9, f"{output_dir}/task_9_health_profile.json")

        print("\n[10/10] Generating Task 10...")
        task10 = self.generate_task10_health_tradeoff(samples_per_task)
        task_results['Task_10_Health_Tradeoff'] = task10
        all_instructions.extend(task10)
        if save_individual:
            self.save_task_to_json(task10, f"{output_dir}/task_10_health_tradeoff.json")

        print("\n" + "=" * 80)
        print(f"TOTAL INSTRUCTIONS GENERATED: {len(all_instructions)}")
        print("=" * 80)

        return all_instructions, task_results

    def save_task_to_json(self, instructions, output_path):
        """Save individual task instructions to JSON file."""
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, indent=2, ensure_ascii=False)
        print(f"  → Saved {len(instructions)} instructions to {output_path}")

    def save_to_json(self, instructions, output_path):
        """Save instructions to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved {len(instructions)} total instructions to {output_path}")

    def save_statistics(self, instructions, stats_path):
        """Save dataset statistics."""
        task_counts = defaultdict(int)
        for inst in instructions:
            task_counts[inst['task_type']] += 1

        stats = {
            "total_instructions": len(instructions),
            "task_breakdown": dict(task_counts),
            "generation_date": datetime.now().isoformat()
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f"\nTask breakdown:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")


# Main execution
if __name__ == "__main__":
    # Paths
    recipes_path = "Recipes.csv"
    reviews_path = "Reviews.csv"
    output_path = "instruction_dataset.json"  # Combined file
    stats_path = "dataset_statistics.json"
    output_dir = "."  # Individual task files saved in same directory

    # Generate dataset with 100K samples per task (1M total)
    print("\n" + "=" * 80)
    print("TARGET: 100,000 samples per task × 10 tasks = 1,000,000 total instructions")
    print("Each task will be saved as a separate JSON file")
    print("=" * 80)

    generator = InstructionDatasetGenerator(recipes_path, reviews_path)
    instructions, task_results = generator.generate_all_tasks(
        samples_per_task=100000,
        save_individual=True,
        output_dir=output_dir
    )

    # Save combined results
    print("\n" + "=" * 80)
    print("SAVING COMBINED DATASET")
    print("=" * 80)
    generator.save_to_json(instructions, output_path)
    generator.save_statistics(instructions, stats_path)

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"✓ Total unique instructions generated: {len(instructions):,}")
    print(f"✓ Total duplicates removed: {len(generator.seen_instruction_hashes) - len(instructions):,}")
    print(f"\nGenerated files:")
    print(f"  1. {output_path} (combined dataset - all 10 tasks)")
    print(f"  2. {stats_path} (statistics)")
    print(f"  3-12. task_1_preference_simulation.json through task_10_health_tradeoff.json")
    print("\nIndividual task files:")
    for i, (task_name, task_data) in enumerate(task_results.items(), 1):
        print(f"  Task {i}: {len(task_data):,} instructions")
    print("=" * 80)
