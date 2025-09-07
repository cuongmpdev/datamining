# Data Mining App - AI Agent Guidelines

## Architecture Overview
This is a Flask web application implementing rough set theory analysis for data mining. The app uses a modular blueprint pattern where:

- `app.py` serves as the main application entry point
- Feature modules live in `pages/` directory (e.g., `pages/reduct/`)
- Each module contains its own templates in `module/templates/`
- Global templates are in `templates/`

## Key Patterns

### Blueprint Structure
Register new features as blueprints in `app.py`:
```python
from pages.reduct.reduct import bp as reduct_bp
app.register_blueprint(reduct_bp)
```

Create blueprint modules with this pattern:
```python
bp = Blueprint(
    'reduct',
    __name__,
    url_prefix='/reduct',
    template_folder='templates'
)
```

### Data Processing
Use pandas DataFrames for data manipulation. The app processes rough set theory concepts:

- **Equivalence classes**: Group objects by attribute values using `df.groupby()`
- **Lower/Upper approximations**: Calculate set approximations with custom functions
- **Alpha coefficient**: Compute accuracy measures as `len(lower) / len(upper)`

### UI Conventions
- Use Bootstrap 5.3.0 for styling (CDN links in templates)
- Vietnamese language throughout UI text and comments
- Form inputs expect comma-separated values (e.g., "O1,O3,O4" for object sets)
- Display pandas DataFrames as HTML tables using `df.to_html(index=False)`

### Development Workflow
- Run with `python app.py` (debug mode enabled by default)
- Access main app at `/`, rough set analysis at `/reduct`
- No requirements.txt - install dependencies manually: `pip install flask pandas`

## File Organization
- `pages/reduct/reduct.py`: Rough set analysis logic and routes
- `pages/reduct/templates/`: Module-specific HTML templates
- `templates/home.html`: Main navigation page
- Data processing functions are defined within route handlers

## Common Tasks
- Adding new data mining algorithms: Create new blueprint in `pages/`
- Modifying data display: Update templates with `{{ table | safe }}` for DataFrame rendering
- Extending calculations: Add functions following the `lower_upper_approximation()` pattern</content>
<parameter name="filePath">/Users/ngocdevv/datamining/.github/copilot-instructions.md
