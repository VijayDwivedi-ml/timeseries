# svm_explainer.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# --- Main App Structure ---
st.set_page_config(
    page_title="SVM Explainer: A Graduate Student's Guide",
    page_icon="🧠",
    layout="wide"
)

st.title("Module III: Support Vector Machine 🧠")
st.markdown("---")

st.sidebar.title("Topic List")
page_selection = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "1. Decision Boundaries",
        "2. Maximum Margin Hyperplanes",
        "3. Structural Risk Minimization",
        "4. Linear SVM - Separable Case",
        "5. Linear SVM - Non-Separable Case",
        "6. Kernel Function & Kernel Trick",
        "7. Kernel Hilbert Space",
        "8. Model Evaluation"
    ]
)

# --- Page Content Functions ---

def page_introduction():
    st.header("Welcome to the SVM Explainer!")
    st.write(
        r"""
        Greetings, early graduate students! I'm Vijay, your mentor for this module, and we're about to embark on an exciting journey into the world of Support Vector Machines (SVMs).
        SVMs are a powerful class of supervised learning models used for classification and regression tasks.
        They are particularly known for their ability to handle complex, high-dimensional data and their strong theoretical foundations rooted in statistical learning theory.

        We'll demystify SVMs, starting from their fundamental principles
        to the clever 'Kernel Trick' that allows them to tackle even the most non-linear problems.

        Use the navigation panel on the left to explore different topics. Each section will provide:
        * Clear, plain English explanations.
        * Simple, intuitive examples.
        * Python code to illustrate concepts.
        * Thought-provoking questions and answers to solidify your understanding.

        Ready to dive in? Let's begin our exploration of SVMs!
        """
    )
    
    st.image(r"https://media.geeksforgeeks.org/wp-content/uploads/20201211181531/Capture.JPG",
                caption="Visualizing the core idea: separating data with a maximal margin.", width=400)

    st.subheader("Why are SVMs important for a graduate student?")
    st.write(
        r"""
        As a graduate student in Machine Learning, understanding SVMs is crucial for several reasons:
        * **Robustness**: They are less prone to overfitting than some other models, especially in high-dimensional spaces.
        * **Strong Theoretical Foundation**: Built on the principles of Structural Risk Minimization (which we'll discuss),
            they offer a strong guarantee of generalization performance.
        * **Versatility**: Capable of handling both linear and non-linear classification/regression problems.
        * **Foundation for Advanced Concepts**: Many advanced machine learning techniques build upon the ideas introduced by SVMs and kernel methods.
        """
    )

    st.subheader("Quick Quiz: True or False?")
    st.write(r"SVMs are primarily used for unsupervised learning tasks.")
    if st.button("Reveal Answer"):
        st.write("False. SVMs are supervised learning models, primarily used for classification and regression.")

def page_decision_boundaries():
    st.header("1. Decision Boundaries for Support Vector Machine")
    st.write(
        r"""
        At its heart, a classification model aims to draw a line (or a curve, or a hyperplane in higher dimensions)
        that separates different classes of data points. This separator is what we call a **decision boundary**.
        Think of it as a fence that divides your garden into different sections for different types of plants.

        For an SVM, the decision boundary is a **hyperplane**. In 2D, it's a line; in 3D, it's a plane;
        and in higher dimensions, it's an N-1 dimensional subspace.
        The goal of the SVM is to find the *best* such hyperplane that separates the data.
        """
    )

    st.subheader("Simple Example: Points in a Plane")
    st.write(
        """
        Imagine we have two types of data points, 'A' (red circles) and 'B' (blue squares), scattered on a 2D plane.
        Our task is to draw a line that best separates these two groups.
        """
    )

    # Generate some simple separable data
    X_sep, y_sep = datasets.make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=0.8)
    # Make classes 0 and 1
    y_sep = np.where(y_sep == 0, -1, 1) # SVMs often use -1 and 1 for classes

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_sep[:, 0], X_sep[:, 1], c=y_sep, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    ax.set_title("Simple Separable Data")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    plt.close(fig) # Prevent Matplotlib from showing the plot outside of Streamlit
    st.pyplot(fig)

    st.write(
        """
        The decision boundary for an SVM is a line that divides the data points such that points on one side
        belong to one class and points on the other side belong to the other class.
        The magic of SVM is *how* it chooses this line, which we'll explore next with 'Maximum Margin Hyperplanes'.
        """
    )

    st.subheader("Concept Check:")
    st.write("If we have a classification problem with 3 features, what would the decision boundary be called?")
    if st.button("Show Answer", key="db_q1"):
        st.write("A **hyperplane** in a 3-dimensional space.")

def page_max_margin_hyperplanes():
    st.header("2. Maximum Margin Hyperplanes")
    st.write(
        """
        This is where SVMs truly shine! Instead of just finding *any* line that separates the classes,
        SVMs look for the one that has the **largest possible margin** between the two classes.
        The 'margin' is the distance between the decision boundary and the closest data point from either class.
        These closest data points are called **support vectors**.
        """
    )
    st.image(r"https://media.geeksforgeeks.org/wp-content/uploads/20210309180309/SVM-660x371.jpg",
             caption="The separating hyperplane and the margin, defined by support vectors.", width=600)
    st.write(
        r"""
        Why maximize the margin?
        Intuitively, a larger margin means a more robust and generalized model. If your decision boundary is
        too close to some data points, a tiny bit of noise or a new, slightly different data point could
        cause it to be misclassified. A wide margin provides a "cushion" or "safety zone," making the
        classification more reliable for unseen data. This relates directly to the idea of
        **generalization** – how well your model performs on new, unseen data.

        Mathematically, if our decision boundary is defined by the equation $w \cdot x - b = 0$,
        then the margin is inversely proportional to $||w||$ (the norm of the weight vector).
        Maximizing the margin is equivalent to minimizing $||w||$ subject to certain constraints.
        """
    )

    st.subheader("Python Example: Visualizing the Margin")
    st.write(
        """
        Let's train a simple Linear SVM and visualize its decision boundary and the margin.
        The points that lie on the margin (or within it in the non-separable case) are the **support vectors**.
        """
    )

    # Generate some linearly separable data
    X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=6, cluster_std=0.6)
    # Adjust y for SVC
    y = np.where(y == 0, -1, 1)

    # Train a linear SVM
    svm_clf = SVC(kernel='linear', C=1000) # C is a regularization parameter, large C for hard margin
    svm_clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot the decision boundary and the margins
    disp = DecisionBoundaryDisplay.from_estimator(
        svm_clf, X, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax,
        levels=[-1, 0, 1], linestyles=["--", "-", "--"]
    )
    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", s=30, edgecolors="k")

    # Highlight support vectors
    ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1],
               s=150, facecolors='none', edgecolors='green', linewidth=2, label="Support Vectors")

    ax.set_title("SVM with Maximum Margin Hyperplane")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.close(fig)
    st.pyplot(fig)

    st.write(
        """
        Notice the dashed lines: these represent the margin. The points on these dashed lines are the **support vectors**.
        They are the crucial data points that define the decision boundary and the margin. If you remove any other
        data point, the decision boundary wouldn't change, but if you remove a support vector, it might!
        """
    )

    st.subheader("Quick Quiz:")
    st.write("What are the special data points that define the margin in an SVM called?")
    if st.button("Check Answer", key="mmh_q1"):
        st.write("They are called **Support Vectors**.")

def page_structural_risk_minimization():
    st.header("3. Structural Risk Minimization (SRM)")
    st.write(
        """
        SVMs are not just based on an intuitive idea; they have a strong theoretical backing in
        **Statistical Learning Theory**, particularly through the principle of **Structural Risk Minimization (SRM)**.
        Developed by Vapnik and Chervonenkis (the "VC" in VC-dimension), SRM provides a framework for selecting
        a model that generalizes well to unseen data.

        In essence, machine learning models face a trade-off:
        1.  **Empirical Risk (Training Error)**: How well the model performs on the data it was trained on.
        2.  **Generalization Risk (True Error)**: How well the model performs on unseen data.

        A model that minimizes empirical risk might simply be memorizing the training data, leading to
        **overfitting**. SRM aims to minimize the generalization risk by balancing the empirical risk with a
        measure of model complexity.

        The SRM principle states that for a given set of models (or "structures"), one should choose the model
        that minimizes an upper bound on the generalization error. This upper bound depends on two factors:
        * The empirical error (how well it performs on the training data).
        * The complexity of the model, often measured by its **VC-dimension**.

        For SVMs, maximizing the margin directly relates to minimizing the VC-dimension, and thus,
        minimizing the structural risk. A larger margin implies a simpler model in terms of its capacity
        to fit arbitrary data, which helps in better generalization.
        """
    )
    st.image(r"https://fderyckel.github.io/machinelearningwithr/otherpics/ModelComplexity_TotalError.png",
             caption="Illustration of the trade-off between training error and model complexity. SRM seeks the optimal balance.", width=600)
    st.write(
        """
        **Key takeaway**: SVMs are not just about finding a separating hyperplane; they are about finding
        the *simplest* separating hyperplane that generalizes well, thanks to the SRM principle.
        This theoretical foundation makes SVMs robust and provides performance guarantees, especially in
        high-dimensional spaces with limited data.
        """
    )

    st.subheader("Concept Check:")
    st.write("What are the two types of risks that Structural Risk Minimization tries to balance?")
    if st.button("Answer", key="srm_q1"):
        st.write("SRM tries to balance **Empirical Risk (training error)** and **Generalization Risk (true error)**, taking into account model complexity.")

def page_linear_svm_separable():
    st.header("4. Linear SVM - Separable Case")
    st.write(
        """
        Let's start with the simplest scenario: when your data can be perfectly separated by a single straight
        line (in 2D) or a flat hyperplane (in higher dimensions). This is known as the **linearly separable case**.
        """
    )
    st.write(
        r"""
        In this ideal situation, the SVM's objective is to find the unique hyperplane that maximizes the margin
        between the two classes. There are no data points that fall between the margin boundaries or on the wrong side of the hyperplane.

        Mathematically, for a linearly separable dataset, the SVM tries to find weights $w$ and bias $b$ such that for all data points $(x_i, y_i)$:
        * $w \cdot x_i - b \ge 1$ if $y_i = 1$
        * $w \cdot x_i - b \le -1$ if $y_i = -1$

        And it minimizes $||w||$ to maximize the margin.
        """
    )

    st.subheader("Python Example: Perfectly Separable Data")
    st.write(
        """
        We'll generate some data that is clearly separable and train a linear SVM on it.
        You'll see a clear decision boundary with a wide margin.
        """
    )

    # Generate separable data
    X_sep, y_sep = datasets.make_blobs(n_samples=80, centers=2, random_state=42, cluster_std=0.6)
    y_sep = np.where(y_sep == 0, -1, 1) # Adjust labels for SVM

    # Train a Linear SVM
    svm_sep_clf = SVC(kernel='linear', C=1000) # Large C means hard margin
    svm_sep_clf.fit(X_sep, y_sep)

    fig, ax = plt.subplots(figsize=(10, 7))
    disp = DecisionBoundaryDisplay.from_estimator(
        svm_sep_clf, X_sep, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax,
        levels=[-1, 0, 1], linestyles=["--", "-", "--"]
    )
    ax.scatter(X_sep[:, 0], X_sep[:, 1], c=y_sep, cmap="RdBu", s=30, edgecolors="k")
    ax.scatter(svm_sep_clf.support_vectors_[:, 0], svm_sep_clf.support_vectors_[:, 1],
               s=150, facecolors='none', edgecolors='green', linewidth=2, label="Support Vectors")
    ax.set_title("Linear SVM: Linearly Separable Case")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.close(fig)
    st.pyplot(fig)

    st.write(
        """
        As you can observe, all data points are correctly classified, and there's a clear margin.
        The support vectors (highlighted in green) are the points closest to the decision boundary,
        defining its position and the width of the margin.
        """
    )

    st.subheader("Key Question:")
    st.write("In a perfectly linearly separable case, what would happen if we used a very small value for the regularization parameter C in SVC?")
    if st.button("Hint / Answer", key="linsep_q1"):
        st.write(
            """
            In `SVC(kernel='linear', C=...)`, `C` is a regularization parameter.
            * A **large C** (like 1000) means the SVM tries very hard to avoid misclassifying any training point,
                leading to a "hard margin". If the data is truly separable, this is fine.
            * A **small C** allows for some misclassifications or points within the margin (a "soft margin").
                In a perfectly separable case, a very small C might lead to a wider margin but potentially misclassify
                some points even if a perfect separation exists, as it prioritizes a wider margin over perfect separation on training data.
                However, for truly separable data, a hard margin (large C) is usually preferred.
            """
        )

def page_linear_svm_non_separable():
    st.header("5. Linear SVM - Non-Separable Case")
    st.write(
        """
        Real-world data is rarely perfectly linearly separable. Often, your data points for different classes
        might overlap, making it impossible to draw a single straight line that separates them without
        any errors. This is the **linearly non-separable case**.
        """
    )
    st.write(
        r"""
        So, what does SVM do here? It introduces the concept of a **soft margin**.
        Instead of strictly enforcing that all points must be outside the margin,
        a soft margin allows for some misclassifications or points to lie within the margin.
        The trade-off between maximizing the margin and minimizing misclassifications is controlled by a
        hyperparameter called **C (Cost parameter)**.

        * **Small C**: Allows for a wider margin but more misclassifications (higher bias, lower variance).
        * **Large C**: Enforces a narrower margin but fewer misclassifications (lower bias, higher variance).

        The misclassified points or points within the margin are penalized using **slack variables** ($\xi_i$).
        The objective function then includes a term for these slack variables, weighted by C.
        The goal is to minimize $||w||^2 + C \sum \xi_i$.
        """
    )

    st.subheader("Python Example: Non-Separable Data with Soft Margin")
    st.write(
        """
        Let's generate some data with overlap and observe how the SVM handles it with a soft margin.
        We'll experiment with different `C` values.
        """
    )

    # Generate non-separable data
    X_non_sep, y_non_sep = datasets.make_blobs(n_samples=100, centers=2, random_state=8, cluster_std=1.2)
    y_non_sep = np.where(y_non_sep == 0, -1, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("C = 0.1 (Softer Margin, More Tolerant)")
        svm_non_sep_clf_low_C = SVC(kernel='linear', C=0.1)
        svm_non_sep_clf_low_C.fit(X_non_sep, y_non_sep)

        fig_low_c, ax_low_c = plt.subplots(figsize=(8, 6))
        disp_low_c = DecisionBoundaryDisplay.from_estimator(
            svm_non_sep_clf_low_C, X_non_sep, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax_low_c,
            levels=[-1, 0, 1], linestyles=["--", "-", "--"]
        )
        ax_low_c.scatter(X_non_sep[:, 0], X_non_sep[:, 1], c=y_non_sep, cmap="RdBu", s=30, edgecolors="k")
        ax_low_c.scatter(svm_non_sep_clf_low_C.support_vectors_[:, 0], svm_non_sep_clf_low_C.support_vectors_[:, 1],
                         s=150, facecolors='none', edgecolors='green', linewidth=2, label="Support Vectors")
        ax_low_c.set_title("Linear SVM: Non-Separable (C=0.1)")
        ax_low_c.set_xlabel("Feature 1")
        ax_low_c.set_ylabel("Feature 2")
        ax_low_c.legend()
        plt.close(fig_low_c)
        st.pyplot(fig_low_c)
        st.write("Notice how the margin is wider, allowing more points to be within or misclassified, aiming for better generalization.")

    with col2:
        st.subheader("C = 100 (Harder Margin, Less Tolerant)")
        svm_non_sep_clf_high_C = SVC(kernel='linear', C=100)
        svm_non_sep_clf_high_C.fit(X_non_sep, y_non_sep)

        fig_high_c, ax_high_c = plt.subplots(figsize=(8, 6))
        disp_high_c = DecisionBoundaryDisplay.from_estimator(
            svm_non_sep_clf_high_C, X_non_sep, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax_high_c,
            levels=[-1, 0, 1], linestyles=["--", "-", "--"]
        )
        ax_high_c.scatter(X_non_sep[:, 0], X_non_sep[:, 1], c=y_non_sep, cmap="RdBu", s=30, edgecolors="k")
        ax_high_c.scatter(svm_non_sep_clf_high_C.support_vectors_[:, 0], svm_non_sep_clf_high_C.support_vectors_[:, 1],
                          s=150, facecolors='none', edgecolors='green', linewidth=2, label="Support Vectors")
        ax_high_c.set_title("Linear SVM: Non-Separable (C=100)")
        ax_high_c.set_xlabel("Feature 1")
        ax_high_c.set_ylabel("Feature 2")
        ax_high_c.legend()
        plt.close(fig_high_c)
        st.pyplot(fig_high_c)
        st.write("Here, the margin is narrower, trying harder to correctly classify training points, potentially leading to overfitting.")

    st.write(
        """
        The choice of `C` is a crucial hyperparameter that needs to be tuned (e.g., using cross-validation)
        to achieve the best generalization performance on unseen data. It's a balance between fitting the training data well and preventing overfitting.
        """
    )

    st.subheader("Quick Check:")
    st.write("What happens to the margin if you increase the value of C in a non-separable case?")
    if st.button("Answer", key="linnonsep_q1"):
        st.write("Increasing C (making it a 'harder' margin) generally leads to a **narrower margin** and fewer training errors, but can increase the risk of overfitting.")

def page_kernel_function_trick():
    st.header("6. Kernel Function & Kernel Trick")
    st.write(
        """
        So far, we've only talked about linear decision boundaries. But what if your data isn't linearly separable
        at all? Imagine a dataset where one class forms a circle within another class. A straight line simply won't cut it!

        This is where the **Kernel Trick** comes to our rescue!
        The core idea is to transform your data from its original, low-dimensional space into a much higher-dimensional
        space, where it *might* become linearly separable. Once it's linearly separable in this higher dimension,
        a standard linear SVM can be applied.
        """
    )
    st.image(r"https://miro.medium.com/v2/resize:fit:838/1*gXvhD4IomaC9Jb37tzDUVg.png",
             caption="The Kernel Trick: Projecting non-linear data to a higher dimension where it becomes linear.", width=600)
    st.write(
        r"""
        ### The "Trick" Part:
        The "trick" is that we don't actually need to calculate the coordinates of the data points in this
        high-dimensional space explicitly. Instead, SVMs only need the **dot product** of the feature vectors.
        A **Kernel Function** (or simply "Kernel") is a function that calculates this dot product in the
        higher-dimensional space *without explicitly performing the transformation*.

        This saves an enormous amount of computational power, especially when the transformed space is
        very high-dimensional (even infinite-dimensional!).

        ### Common Kernel Functions:
        1.  **Linear Kernel**: $K(x_i, x_j) = x_i \cdot x_j$ (This is what we've used for linear SVMs)
        2.  **Polynomial Kernel**: $K(x_i, x_j) = (\gamma x_i \cdot x_j + r)^d$
            * $d$: degree of the polynomial
            * $\gamma$: scaling factor
            * $r$: constant term
        3.  **Radial Basis Function (RBF) / Gaussian Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
            * $\gamma$: controls the influence of individual training samples. A small $\gamma$ means a large radius of influence, a large $\gamma$ means a small radius.
        4.  **Sigmoid Kernel**: $K(x_i, x_j) = \tanh(\gamma x_i \cdot x_j + r)$

        The RBF kernel is one of the most widely used and effective kernels, capable of handling complex non-linear relationships.
        """
    )

    st.subheader("Python Example: Non-Linear Data with RBF Kernel")
    st.write(
        """
        Let's create some non-linearly separable data (e.g., concentric circles) and see how a linear SVM
        fails, but an RBF kernel SVM succeeds.
        """
    )

    # Generate non-linear data (circles)
    X_circles, y_circles = datasets.make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    y_circles = np.where(y_circles == 0, -1, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Linear SVM (Fails)")
        svm_linear_fail = SVC(kernel='linear', C=1)
        svm_linear_fail.fit(X_circles, y_circles)

        fig_linear, ax_linear = plt.subplots(figsize=(8, 6))
        disp_linear = DecisionBoundaryDisplay.from_estimator(
            svm_linear_fail, X_circles, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax_linear,
            levels=[-1, 0, 1], linestyles=["--", "-", "--"]
        )
        ax_linear.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="RdBu", s=30, edgecolors="k")
        ax_linear.set_title("Linear SVM on Non-Linear Data")
        ax_linear.set_xlabel("Feature 1")
        ax_linear.set_ylabel("Feature 2")
        plt.close(fig_linear)
        st.pyplot(fig_linear)
        st.write("A linear SVM struggles to separate the concentric circles.")

    with col2:
        st.subheader("RBF Kernel SVM (Succeeds)")
        svm_rbf_success = SVC(kernel='rbf', C=1, gamma='scale') # gamma='scale' uses 1 / (n_features * X.var())
        svm_rbf_success.fit(X_circles, y_circles)

        fig_rbf, ax_rbf = plt.subplots(figsize=(8, 6))
        disp_rbf = DecisionBoundaryDisplay.from_estimator(
            svm_rbf_success, X_circles, plot_method="contour", cmap="RdBu", alpha=0.8, ax=ax_rbf,
            levels=[-1, 0, 1], linestyles=["--", "-", "--"]
        )
        ax_rbf.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="RdBu", s=30, edgecolors="k")
        ax_rbf.set_title("RBF Kernel SVM on Non-Linear Data")
        ax_rbf.set_xlabel("Feature 1")
        ax_rbf.set_ylabel("Feature 2")
        plt.close(fig_rbf)
        st.pyplot(fig_rbf)
        st.write("The RBF kernel successfully finds a non-linear decision boundary.")

    st.write(
        """
        This is the power of the Kernel Trick! It allows us to apply linear classification methods to
        solve non-linear problems by implicitly mapping data to a higher-dimensional feature space.
        """
    )

    st.subheader("Common Misconception:")
    st.write("Does the Kernel Trick actually transform the data into a higher dimensional space?")
    if st.button("Clarify"):
        st.write(
            """
            No, not explicitly! That's the "trick". The kernel function calculates the dot product
            *as if* the data were transformed into a higher-dimensional space, without ever
            actually performing the expensive transformation itself. It operates directly on the
            original feature vectors.
            """
        )

def page_kernel_hilbert_space():
    st.header("7. Kernel Hilbert Space")
    st.write(
        r"""
        To truly appreciate the Kernel Trick, let's briefly touch upon the theoretical concept behind it:
        the **Reproducing Kernel Hilbert Space (RKHS)**. Don't let the name intimidate you;
        it's a beautiful mathematical framework that makes the Kernel Trick work.

        When we apply a kernel function $K(x_i, x_j)$ to our data points $x_i$ and $x_j$, we are implicitly mapping these
        points from our original input space ($X$) to a potentially very high-dimensional (even infinite-dimensional)
        feature space ($\mathcal{F}$). This new space, equipped with an inner product derived from the kernel, is a
        Hilbert space, and it's called a Reproducing Kernel Hilbert Space.

        The key property of an RKHS is that the inner product in this feature space can be computed using the kernel function:
        $$ \langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{F}} = K(x_i, x_j) $$
        where $\phi$ is the (often implicit) mapping from the input space to the feature space.

        ### Why is this important?
        * **Linearity in High Dimensions**: In the RKHS, even complex non-linear relationships in the original space become linear.
            This means the SVM can find a linear decision boundary in this higher-dimensional space, which translates
            back to a non-linear decision boundary in the original space.
        * **Computational Efficiency**: As discussed with the Kernel Trick, we never actually need to compute $\phi(x)$
            or work with these high-dimensional vectors directly. All calculations only involve the kernel function $K$.
        * **Theoretical Guarantees**: RKHS provides the mathematical rigor for why kernel methods work and generalize well.
            It ensures that the function found by the SVM (which lives in the RKHS) is well-behaved and unique.

        Think of it this way: the RKHS is the "playground" where the linear SVM operates, even though we, as users,
        only interact with the data in its original, simpler form through the kernel function.
        """
    )
    st.image(r"https://image.slideserve.com/338368/hilbert-spaces27-l.jpg",
             caption="An analogy for Hilbert space: a complete inner product space.", width=600)

    st.subheader("Analogy:")
    st.write(
        """
        Imagine you have two friends, Alice and Bob, who are trying to decide if two complex paintings are "similar".
        They can't easily quantify this similarity by just looking at the 2D canvas (original space).
        However, if they could transform their understanding of the paintings into a higher-dimensional space of
        "artistic style," "color palette," "brush stroke density," etc., then suddenly, calculating similarity
        (dot product) in *that* space becomes easier.

        The Kernel function is like a special tool that allows them to instantly calculate this "similarity score"
        in the "artistic style" space *without ever having to explicitly list out all the style features*.
        The RKHS is that abstract "artistic style" space where the similarity truly makes sense.
        """
    )

    st.subheader("Your Turn:")
    st.write("What is the primary benefit of working implicitly in a Kernel Hilbert Space via the Kernel Trick?")
    if st.button("See Explanation", key="khs_q1"):
        st.write(
            """
            The primary benefit is **computational efficiency and the ability to find non-linear decision boundaries**
            in the original feature space. It allows us to apply linear classification methods to inherently non-linear problems
            without the explicit, computationally expensive transformation to a high-dimensional space.
            """
        )

def page_model_evaluation():
    st.header("8. Model Evaluation")
    st.write(
        """
        After training your SVM, how do you know if it's actually doing a good job? This is where **model evaluation** comes in.
        It's crucial to assess your model's performance on **unseen data** to get an unbiased estimate of its generalization ability.
        """
    )

    st.subheader("Key Metrics for Classification (SVMs are often used for classification):")
    st.write(
        r"""
        1.  **Accuracy**: The proportion of correctly classified instances.
            $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
            * **When to use**: Simple and intuitive.
            * **Caveat**: Can be misleading with imbalanced datasets. If 95% of your data is Class A, a model predicting Class A for everything will have 95% accuracy but be useless.

        2.  **Precision**: Of all instances predicted as positive, how many are actually positive?
            $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$
            * **When to use**: When the cost of a False Positive is high (e.g., medical diagnosis: don't diagnose someone with a disease they don't have).

        3.  **Recall (Sensitivity)**: Of all actual positive instances, how many did the model correctly identify?
            $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
            * **When to use**: When the cost of a False Negative is high (e.g., fraud detection: don't miss actual fraud).

        4.  **F1-Score**: The harmonic mean of Precision and Recall. It's a good metric for imbalanced datasets.
            $$ \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
            * **When to use**: When you need a balance between Precision and Recall, especially with uneven class distribution.

        5.  **Confusion Matrix**: A table that summarizes the performance of a classification algorithm.
            It shows the number of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

        6.  **ROC Curve and AUC (Area Under the Curve)**:
            * **ROC Curve**: Plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
            * **AUC**: Represents the degree or measure of separability. The higher the AUC, the better the model is at distinguishing between classes.

        """
    )
    st.image(r"https://databasecamp.de/wp-content/uploads/confusion-matrix-structure-e1658123198194-1024x537.png",
             caption="A typical Confusion Matrix structure.", width=600)

    st.subheader("Cross-Validation:")
    st.write(
        """
        To get a more robust estimate of your model's performance and to tune hyperparameters (like `C` and `gamma` for SVMs),
        we use **cross-validation**. The most common type is K-Fold Cross-Validation:
        1.  Divide the dataset into K equally sized "folds".
        2.  Train the model K times. In each iteration, use K-1 folds for training and the remaining 1 fold for testing.
        3.  The performance metrics are then averaged across all K iterations.

        This helps reduce the variance of the performance estimate and provides a more reliable measure of how well the model
        will generalize to unseen data.
        """
    )

    st.subheader("Python Example: Evaluating an SVM")
    st.write(
        """
        Let's train an SVM and evaluate its performance using common metrics.
        """
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
    import pandas as pd
    import seaborn as sns

    # Generate sample data
    X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    y = np.where(y == 0, -1, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train SVM
    svm_eval_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True) # probability=True for ROC curve
    svm_eval_clf.fit(X_train, y_train)
    y_pred = svm_eval_clf.predict(X_test)

    st.subheader("Performance Metrics:")
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col_met2.metric("Precision", f"{precision_score(y_test, y_pred, pos_label=1):.2f}") # Specify pos_label for binary
    col_met3.metric("Recall", f"{recall_score(y_test, y_pred, pos_label=1):.2f}")
    col_met4.metric("F1-Score", f"{f1_score(y_test, y_pred, pos_label=1):.2f}")

    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1]) # Ensure labels order
    cm_df = pd.DataFrame(cm, index=['Actual -1', 'Actual 1'], columns=['Predicted -1', 'Predicted 1'])
    st.dataframe(cm_df)

    st.subheader("ROC Curve:")
    y_scores = svm_eval_clf.predict_proba(X_test)[:, 1] # Probability of the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    plt.close(fig_roc)
    st.pyplot(fig_roc)

    st.write(
        """
        These metrics give you a comprehensive understanding of how well your SVM model is performing
        on unseen data, which is the ultimate test of any machine learning model.
        """
    )

    st.subheader("Quick Question:")
    st.write("Why is it important to use a separate test set (or cross-validation) for model evaluation instead of just evaluating on the training data?")
    if st.button("Think and Answer", key="me_q1"):
        st.write(
            """
            Evaluating on the training data gives an overly optimistic view of performance and doesn't tell you how well the model will generalize to new, unseen data.
            A separate test set (or cross-validation) provides an unbiased estimate of the model's generalization ability, helping to detect overfitting.
            """
        )

# --- Streamlit Page Routing ---
if page_selection == "Introduction":
    page_introduction()
elif page_selection == "1. Decision Boundaries":
    page_decision_boundaries()
elif page_selection == "2. Maximum Margin Hyperplanes":
    page_max_margin_hyperplanes()
elif page_selection == "3. Structural Risk Minimization":
    page_structural_risk_minimization()
elif page_selection == "4. Linear SVM - Separable Case":
    page_linear_svm_separable()
elif page_selection == "5. Linear SVM - Non-Separable Case":
    page_linear_svm_non_separable()
elif page_selection == "6. Kernel Function & Kernel Trick":
    page_kernel_function_trick()
elif page_selection == "7. Kernel Hilbert Space":
    page_kernel_hilbert_space()
elif page_selection == "8. Model Evaluation":
    page_model_evaluation()
