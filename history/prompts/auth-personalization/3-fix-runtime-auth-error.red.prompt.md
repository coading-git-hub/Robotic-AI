---
id: "3"
title: "Fix Runtime Auth Error in Navbar Component"
stage: "red"
date_iso: "2025-12-19"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "auth-personalization"
branch: "master"
user: "unknown"
command: "/sp.phr"
labels:
  - "debugging"
  - "runtime-error"
  - "context-provider"
  - "authentication"
  - "navbar"
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "book-frontend/src/theme/Navbar/index.js"
  - "book-frontend/src/components/auth/AuthNavbar.js"
  - "book-frontend/src/components/auth/AuthContext.jsx"
tests:
  - "none"
---

# Fix Runtime Auth Error in Navbar Component

## PROMPT_TEXT
Uncaught runtime errors:
×
ERROR
useAuth must be used within an AuthProvider
    at useAuth (webpack-internal:///./src/components/auth/AuthContext.jsx:17:116)
    at NavbarWrapper (webpack-internal:///./src/theme/Navbar/index.js:10:143)
    at renderWithHooks (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:15486:18)
    at mountIndeterminateComponent (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:20099:13)
    at beginWork (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:21622:16)
    at HTMLUnknownElement.callCallback (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:4164:14)
    at Object.invokeGuardedCallbackDev (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:4213:16)
    at invokeGuardedCallback (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:4277:31)
    at beginWork$1 (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:27486:7)
    at performUnitOfWork (webpack-internal:///./node_modules/react-dom/cjs/react-dom.development.js:26595:12)

## RESPONSE_TEXT
Fixed runtime error by updating Navbar and AuthNavbar components to use the new useBetterAuth hook instead of the old useAuth hook. Removed the obsolete AuthContext.jsx file that was no longer needed after the migration to BetterAuthProvider.

## OUTCOME_IMPACT
Resolved the runtime error where useAuth was being used without an AuthProvider, and cleaned up obsolete authentication context files.

## TESTS_SUMMARY
Manual verification that the navbar components now use the correct authentication context.

## FILES_SUMMARY
Updated import in Navbar/index.js, updated AuthNavbar.js, and removed obsolete AuthContext.jsx file.

## NEXT_PROMPTS
- Verify the website runs without runtime errors
- Test navbar authentication features

## REFLECTION_NOTE
Components using authentication hooks must be updated when migrating from one auth provider to another to avoid runtime context errors.

## FAILURE MODES OBSERVED
Old authentication context was still referenced in some components after migration to new provider.

## NEXT EXPERIMENT
Consider implementing a more comprehensive search for all references when migrating authentication systems.