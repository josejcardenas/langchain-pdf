
# Information taken following the coures
## Description
Here is the follow me solution for the course from udemy https://www.udemy.com/course/chatgpt-and-langchain-the-complete-developers-masterclass/learn/lecture/40275878#questions
Section 7 and up.

## Goal
Follow the langchain inclusion for a solution mixing javascript and python

## Resources
First the starting code here is taken from cestion 68 on section 7 Boilerplate download. Some notes below:
To create the environment I use conda and configure all packages config on environemnt.yml file.  Usually the openai and langchain tools install better using pip


# Previous info on readme
# First Time Setup

## Using Pipenv [Recommended]

```
# Install dependencies
pipenv install

# Create a virtual environment
pipenv shell

# Initialize the database
flask --app app.web init-db

```

## Using Venv [Optional]

These instructions are included if you wish to use venv to manage your evironment and dependencies instead of Pipenv.

```
# Create the venv virtual environment
python -m venv .venv

# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database
flask --app app.web init-db
```

# Running the app [Pipenv]

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below. If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server

Open a new terminal window and create a new virtual environment:

```
pipenv shell
```

Then:

```
inv dev
```

### To run the worker

Open a new terminal window and create a new virtual environment:

```
pipenv shell
```

Then:

```
inv devworker
```

### To run Redis

```
redis-server
```

### To reset the database

Open a new terminal window and create a new virtual environment:

```
pipenv shell
```

Then:

```
flask --app app.web init-db
```

# Running the app [Venv]

_These instructions are included if you wish to use venv to manage your evironment and dependencies instead of Pipenv._

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below. If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server

Open a new terminal window and create a new virtual environment:

```
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```
inv dev
```

### To run the worker

Open a new terminal window and create a new virtual environment:

```
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```
inv devworker
```

### To run Redis

```
redis-server
```

### To reset the database

Open a new terminal window and create a new virtual environment:

```
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```
flask --app app.web init-db
```
