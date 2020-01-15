# System Architecture

## Search

### Algorithm
For Darts and Random search:

```
input: conf_macro, micro_builder
output: final_desc

macro_desc = build_macro(conf_macro)
model_desc = build_desc(macro_desc, micro_builder)
model = build_model(model_desc)
train(model)
final_desc = finalize(model)
```

For PetriDish, we need to add n iteration

```
input: conf_macro, micro_builder, n_search_iter
output: final_desc

macro_desc = build_macro(conf_macro)
for i = 1 to n_search_iter:
    if pre_train_epochs > 0:
        if all nodes non-empty:
            model = build_model(model_desc, restore_state=True)
            train(model)
            macro_desc = finalize(model. include_state=True)
        elif all nodes empty:
            pass because no point in training empty model
        else
            raise exception

    # we have P cells, Q nodes each with 0 edges on i=1 at this point
    # for i > 1, we have P cells, i-1 nodes at this point
    # Petridish micro builder removes 0 edges nodes after i
    # if number of nodes < i, Petridish macro adds nodes
    # assert 0 edges for all nodes for i-1
    # Petridish micro builder adds Petridish op at i
    model_desc = build_desc(macro_desc, micro_builder(i))
    # we have P cells, i node(s) each
    model = build_model(model_desc, restore_state=True)
    arch_train(model)
    macro_desc = final_desc = finalize(model. include_state=True)
    # make sure FinalPetridishOp can+will run in search mode
    # we end with i nodes in each cell for Petridish at this point
```
