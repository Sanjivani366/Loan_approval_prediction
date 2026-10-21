from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('logistic.pkl', 'rb') as f:
  model = pickle.load(f)

# Embedded HTML template with modern styling
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 font-sans text-slate-800">
    <div class="min-h-screen py-10 px-4 sm:px-6 lg:px-8">
        <div class="max-w-3xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100">
            <!-- Header Section -->
            <div class="bg-gradient-to-r from-blue-600 to-indigo-700 py-6 px-8 text-white">
                <h1 class="text-3xl font-bold tracking-tight">Loan Eligibility Portal</h1>
                <p class="mt-1 text-blue-100 text-sm">Fill in your applicant details below to check your instant loan approval status.</p>
            </div>
            
            <!-- Form Section -->
            <form method="POST" class="p-8 space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Gender -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Gender</label>
                        <select name="Gender" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="1">Male</option>
                            <option value="0">Female</option>
                        </select>
                    </div>

                    <!-- Married -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Marital Status</label>
                        <select name="Married" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>

                    <!-- Dependents -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Dependents</label>
                        <select name="Dependents" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3+</option>
                        </select>
                    </div>

                    <!-- Education -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Education</label>
                        <select name="Education" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="1">Graduate</option>
                            <option value="0">Not Graduate</option>
                        </select>
                    </div>

                    <!-- Self Employed -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Self Employed</label>
                        <select name="Self_Employed" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <!-- Credit History -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Credit History</label>
                        <select name="Credit_History" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="1">1.0 (Good Standing)</option>
                            <option value="0">0.0 (Bad/None)</option>
                        </select>
                    </div>

                    <!-- Applicant Income -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Applicant Income ($)</label>
                        <input type="number" name="ApplicantIncome" placeholder="e.g. 5000" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    </div>

                    <!-- Coapplicant Income -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Coapplicant Income ($)</label>
                        <input type="number" name="CoapplicantIncome" placeholder="e.g. 2000" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    </div>

                    <!-- Loan Amount -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Loan Amount (in thousands)</label>
                        <input type="number" name="LoanAmount" placeholder="e.g. 150" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    </div>

                    <!-- Loan Amount Term -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Loan Term (Months)</label>
                        <input type="number" name="Loan_Amount_Term" placeholder="e.g. 360" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none" required>
                    </div>

                    <!-- Property Area -->
                    <div class="md:col-span-2">
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Property Area</label>
                        <select name="Property_Area" class="w-full rounded-lg border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white" required>
                            <option value="2">Urban</option>
                            <option value="1">Semiurban</option>
                            <option value="0">Rural</option>
                        </select>
                    </div>
                </div>

                <!-- Submit Button -->
                <div class="pt-2">
                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg shadow-md transition duration-200">
                        Check Eligibility
                    </button>
                </div>
            </form>

            <!-- Result Box -->
            {% if prediction_text is not none %}
            <div class="px-8 pb-8">
                <div class="p-4 rounded-xl text-center font-semibold text-lg {% if is_approved %}bg-emerald-50 text-emerald-800 border border-emerald-200{% else %}bg-rose-50 text-rose-800 border border-rose-200{% endif %}">
                    {{ prediction_text }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction_text = None
  is_approved = False
  if request.method == 'POST':
    try:
      # Capture and parse form inputs in exact feature order
      gender = float(request.form['Gender'])
      married = float(request.form['Married'])
      dependents = float(request.form['Dependents'])
      education = float(request.form['Education'])
      self_employed = float(request.form['Self_Employed'])
      applicant_income = float(request.form['ApplicantIncome'])
      coapplicant_income = float(request.form['CoapplicantIncome'])
      loan_amount = float(request.form['LoanAmount'])
      loan_amount_term = float(request.form['Loan_Amount_Term'])
      credit_history = float(request.form['Credit_History'])
      property_area = float(request.form['Property_Area'])

      features = np.array([[
          gender,
          married,
          dependents,
          education,
          self_employed,
          applicant_income,
          coapplicant_income,
          loan_amount,
          loan_amount_term,
          credit_history,
          property_area,
      ]])

      prediction = model.predict(features)

      # Interpret model prediction output (1 / 'Y' = Approved)
      if prediction[0] == 1 or prediction[0] == 'Y':
        prediction_text = (
            '🎉 Congratulations! Your Loan Application is APPROVED.'
        )
        is_approved = True
      else:
        prediction_text = (
            '❌ Sorry, your Loan Application is likely to be REJECTED.'
        )
        is_approved = False
    except Exception as e:
      prediction_text = f'Error processing prediction: {str(e)}'
      is_approved = False

  return render_template_string(
      HTML_PAGE, prediction_text=prediction_text, is_approved=is_approved
  )


if __name__ == '__main__':
  app.run(debug=True)
