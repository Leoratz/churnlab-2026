**Question A1:** They are different types of assets and their purposes are different. The CSV contains the raw data, that we are    use to train the model. The training source code contains the instructions used to train the model to train and the fitted model contains the parameters learning during training.
**Question A2:** Churn is the target that the model has to predict. It will be considered data leakage if we include it to the model inputs.

**Tasks B2:**
1.  requires-python = ">=3.11, <3.13" and pandas
2. dependencies = [{ name = "pandas" },] in uv.lock
3. six v1.17.0
4. pyproject.toml defines the project configuration, the uv.lock contains the exact installed versions and .venv/ is where the packages are installed locally.


**C3**
1. Because the model predicts a category, that is if the customer leaves or not.
2. If the classes are imbalanced accuracy can be misleading. A model could predict the majority class most of the time and still have a high accuracy.
3. Because some values can contain spaces or non numeric characters.
4. Because a whitespace " " s still a string, not a missing value like NaN. So we have to remove the whitespace or convert it to a missing value first.
5. The source URL and commit, the hash (SHA-256).

**D2**
| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations |
| Installed environment | Before/after environment records | This does not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |
