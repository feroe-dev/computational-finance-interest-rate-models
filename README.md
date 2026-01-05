# computational-finance-termstructuremodelplugins-project

**Extend and Analyse a Term Structure Model**

The project consists of the improvements of a Monte-Carlo simulation of an Euler-scheme discretization of a *Discrete Term Structure Model* (LIBOR Market Model) and the investigation of properties of the  model.


## Project description

The current version of the project description can be found in the lecture's Moodle's announcements section.


## Importing as Maven project

The project comes with a pre-configured Maven pom.xml file referencing finmath-lib.

- Clone the project using git (`git clone`), then import the project in your favourite IDE as Maven project.


## Notes

### Unit Tests

You may consider adding unit tests to your solution. You find a small stub / sample test in `src/main/test`. You can run unit test from Eclipse or via Maven on the command line

```
mvn clean test
```

(run from the project directory).


### Code Style

If you like to improve your code-style, you may run ``checkstyle`` via the maven command

```
mvn checkstyle:check 
```

(run from the project directory).

Checkstyle will report style issues of your code. If you like to clean up the formatting, you may use *Source -> Clean up...* in Eclipse.


### JavaDoc

To generate JavaDoc run

```
mvn javadoc:javadoc 
```

(if successful, the javadoc will then reside in `target/site/apidocs`). The project is configured to support
LaTeX in JavaDocs (use `\(` and `\)` to open and close a math environment).

Note that JavaDoc is HTML. This implies that an `<` needs to be written as `&lt;` and `>` needs to be written as `>`.

### CI/CD

The project is configured to run Unit Test, JavaDoc, and Checkstyle upon a git push (via GitHub Actions).

## Importing in Eclipse from GitHub

Import this git repository into Eclipse and start working.

- Click on the link to your repository (the link starts with qntlb/computational-finance… )
- Click on “Clone or download” and copy the URL to your clipboard.
- Go to Eclipse and select File -> Import -> Git -> Projects from Git **(with smart import)**.
- Select “Clone URI” and paste the GitHub URL from step 2.
- Select "main", then Next -> Next -> Finish.

Note: If you choose "Projects from Git" without the option "(with smart import)" you may experience that
the project is not imported into Eclipse, but it was successfully checked out via git, i.e. you
find the project files in your local git folder. In that case, you can import the project "as maven project"
(see below).

### Importing in Eclipse (as Maven Project)

If you checked out the git repository manually (`git clone`), then import
the local git folder as Maven Project;

- File -> Import -> Maven -> Existing Maven Projects
- Select the project folder in your *local* git folder.

---

<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>
