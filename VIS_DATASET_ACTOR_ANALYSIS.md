# vis_dataset.py Actor Shuffling Analysis

## How It Works

### Dataset Splitting (RAVDESS class)
1. **Loads all actors**: Finds all `Actor_01` through `Actor_24` directories
2. **Groups by actor**: Groups all items by actor ID
3. **Splits actors**: Splits actors (not items) into train/val/test to avoid data leakage
   - With 24 actors, 80/10/10 split:
     - Train: ~19 actors
     - Val: ~2 actors  
     - Test: ~3 actors
4. **Shuffles items within split**: Items from all actors in that split are shuffled together

### vis_dataset.py Shuffling
1. **Loads a specific split**: `split=args.split` (train/val/test)
2. **Gets all items in that split**: Items from all actors assigned to that split
3. **Shuffles indices**: `order = np.arange(len(ds))` then `rng.shuffle(order)`
4. **Result**: Shuffles across ALL items in the split, which includes items from ALL actors in that split

## Answer: YES, It Shuffles Across All Actors

✅ **The shuffling DOES work across all actors** in the selected split.

However:
- It only shuffles within the **selected split** (train/val/test)
- Each split contains a subset of actors (not all 24)
- With default 80/10/10 split:
  - Train split: ~19 actors
  - Val split: ~2 actors
  - Test split: ~3 actors

## Verification

The updated `vis_dataset.py` now logs:
```
Actors in test split: 3 actors (1-24)
```

This will show:
- How many actors are in the selected split
- Which actor IDs are present
- A warning if fewer than expected actors are found

## Example Output

When you run `make vis MODE=emotion SPLIT=test`, you'll see:
```
Dataset split='test' mode='emotion' size=126
Actors in test split: 3 actors (5-23)
Playback order: shuffled (across all actors in split)
```

This confirms:
- All 3 actors in the test split are included
- Shuffling happens across all items from all actors in that split
- You'll see samples from all actors in the split, randomly interleaved

## Potential Issue

If you want to visualize **all 24 actors** at once, you'd need to:
1. Use `split='train'` (which has ~19 actors), or
2. Create a custom visualization that loads multiple splits, or
3. Modify the script to accept `split='all'` option

But for visualizing a specific split (train/val/test), the shuffling correctly works across all actors in that split.

